import React, {
  useEffect,
  useRef,
  useState,
} from "react";

import "./analysis.css";
import bloodBackground from "./assets/blood-background.mp4";

const API_URL = "http://127.0.0.1:8000/api/analyze";

const CLASS_COLORS = {
  WBC: "#8b5cf6",
  RBC: "#ef4444",
  PLATELETS: "#f59e0b",
  PLATELET: "#f59e0b",
};

/* =========================================================
   HELPERS
========================================================= */

const formatPercent = (value) => {
  if (
    value === null ||
    value === undefined ||
    Number.isNaN(Number(value))
  ) {
    return "0.00%";
  }

  const numeric = Number(value);
  const percent =
    numeric <= 1 ? numeric * 100 : numeric;

  return `${percent.toFixed(2)}%`;
};

const formatConfidence = (value) => {
  if (
    value === null ||
    value === undefined ||
    Number.isNaN(Number(value))
  ) {
    return "0.00%";
  }

  const numeric = Number(value);

  return `${(
    numeric <= 1 ? numeric * 100 : numeric
  ).toFixed(2)}%`;
};

const normalizeSubtype = (value) => {
  if (!value) return "UNCERTAIN";

  return String(value)
    .trim()
    .toUpperCase();
};

const getDecisionMethodLabel = (method) => {
  if (!method) {
    return "3-FOLD DECISION";
  }

  const normalized =
    String(method).toLowerCase();

  if (
    normalized.includes("majority") ||
    normalized.includes("vote")
  ) {
    return "MAJORITY VOTE";
  }

  if (
    normalized.includes("highest") ||
    normalized.includes("probability")
  ) {
    return "HIGHEST PROBABILITY";
  }

  if (normalized.includes("unanimous")) {
    return "UNANIMOUS";
  }

  if (normalized.includes("failed")) {
    return "CLASSIFICATION FAILED";
  }

  return String(method)
    .replaceAll("_", " ")
    .toUpperCase();
};

const getDecisionDescription = (method) => {
  if (!method) {
    return "Final subtype selected from the 3-fold classifier.";
  }

  const normalized =
    String(method).toLowerCase();

  if (
    normalized.includes("majority") ||
    normalized.includes("vote")
  ) {
    return "Two or more folds predicted the same subtype.";
  }

  if (
    normalized.includes("highest") ||
    normalized.includes("probability")
  ) {
    return "All folds differed, so the prediction with the highest probability was selected.";
  }

  if (normalized.includes("unanimous")) {
    return "All three folds predicted the same subtype.";
  }

  if (normalized.includes("failed")) {
    return "The WBC classification could not be completed.";
  }

  return "Final subtype selected by the 3-fold classifier.";
};

const getSubtypeClass = (subtype) => {
  const value =
    normalizeSubtype(subtype);

  if (value === "UNCERTAIN") {
    return "subtype-uncertain";
  }

  if (value === "PC") {
    return "subtype-pc";
  }

  if (value === "BNE") {
    return "subtype-bne";
  }

  if (value === "SNE") {
    return "subtype-sne";
  }

  if (value === "MO") {
    return "subtype-mo";
  }

  if (value === "LY") {
    return "subtype-ly";
  }

  if (value === "EO") {
    return "subtype-eo";
  }

  if (value === "BA") {
    return "subtype-ba";
  }

  return "subtype-default";
};

const getFoldPrediction = (fold) => {
  if (!fold) return "N/A";

  return normalizeSubtype(
    fold.class_name ||
      fold.className ||
      fold.subtype ||
      fold.prediction ||
      fold.label ||
      fold.predicted_class ||
      fold.final_class
  );
};

const getFoldConfidence = (fold) => {
  if (!fold) return 0;

  return (
    fold.confidence ??
    fold.probability ??
    fold.prob ??
    fold.score ??
    fold.max_probability ??
    0
  );
};

const getFoldNumber = (
  fold,
  index
) => {
  if (!fold) return index + 1;

  return (
    fold.fold ??
    fold.fold_number ??
    fold.foldNumber ??
    fold.index ??
    index + 1
  );
};

const getProbabilityEntries = (
  probabilities
) => {
  if (
    !probabilities ||
    typeof probabilities !== "object"
  ) {
    return [];
  }

  return Object.entries(probabilities)
    .map(([name, value]) => ({
      name: normalizeSubtype(name),
      value: Number(value),
    }))
    .filter(
      (item) =>
        !Number.isNaN(item.value)
    )
    .sort(
      (a, b) => b.value - a.value
    );
};


/* =========================================================
   COUNT CARD
========================================================= */

function CountCard({
  title,
  value,
  type,
}) {
  return (
    <div
      className={`count-card count-${type.toLowerCase()}`}
    >
      <div className="count-card-top">
        <span className="count-card-label">
          {title}
        </span>
      </div>

      <div className="count-card-value">
        {value ?? 0}
      </div>
    </div>
  );
}


/* =========================================================
   FOLD CARD
========================================================= */

function FoldCard({
  fold,
  index,
  finalSubtype,
}) {
  const prediction =
    getFoldPrediction(fold);

  const confidence =
    getFoldConfidence(fold);

  const foldNumber =
    getFoldNumber(fold, index);

  const isFinal =
    normalizeSubtype(prediction) ===
    normalizeSubtype(finalSubtype);

  const confidenceValue =
    Number(confidence) <= 1
      ? Number(confidence) * 100
      : Number(confidence);

  return (
    <div
      className={`fold-card ${
        isFinal
          ? "fold-card-selected"
          : ""
      }`}
    >
      <div className="fold-card-header">
        <span className="fold-number">
          FOLD {foldNumber}
        </span>

        {isFinal && (
          <span className="fold-selected-badge">
            SELECTED
          </span>
        )}
      </div>

      <div className="fold-card-prediction">
        <span
          className={getSubtypeClass(
            prediction
          )}
        >
          {prediction}
        </span>
      </div>

      <div className="fold-confidence-label">
        CONFIDENCE
      </div>

      <div className="fold-confidence">
        {formatConfidence(
          confidence
        )}
      </div>

      <div className="fold-progress">
        <div
          className="fold-progress-fill"
          style={{
            width: `${Math.min(
              100,
              Math.max(
                0,
                confidenceValue
              )
            )}%`,
          }}
        />
      </div>
    </div>
  );
}


/* =========================================================
   PROBABILITY PANEL
========================================================= */

function ProbabilityPanel({
  probabilities,
  finalSubtype,
}) {
  const entries =
    getProbabilityEntries(
      probabilities
    );

  if (!entries.length) {
    return null;
  }

  return (
    <div className="probability-panel">
      <div className="section-heading">
        <div>
          <span className="section-kicker">
            CLASS DISTRIBUTION
          </span>

          <h3>
            Subtype Probabilities
          </h3>
        </div>
      </div>

      <div className="probability-list">
        {entries.map((item) => {
          const isFinal =
            item.name ===
            normalizeSubtype(
              finalSubtype
            );

          const width =
            Number(item.value) <= 1
              ? Number(item.value) * 100
              : Number(item.value);

          return (
            <div
              className={`probability-row ${
                isFinal
                  ? "probability-row-selected"
                  : ""
              }`}
              key={item.name}
            >
              <div className="probability-name">
                <span>
                  {item.name}
                </span>

                {isFinal && (
                  <span className="probability-final">
                    FINAL
                  </span>
                )}
              </div>

              <div className="probability-bar-container">
                <div className="probability-bar">
                  <div
                    className="probability-bar-fill"
                    style={{
                      width: `${Math.min(
                        100,
                        Math.max(
                          0,
                          width
                        )
                      )}%`,
                    }}
                  />
                </div>
              </div>

              <div className="probability-value">
                {formatPercent(
                  item.value
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}


/* =========================================================
   DETECTION OVERLAY
========================================================= */

function DetectionOverlay({
  detection,
  imageWidth,
  imageHeight,
}) {
  if (!detection?.bbox) {
    return null;
  }

  if (
    !imageWidth ||
    !imageHeight
  ) {
    return null;
  }

  const bbox =
    detection.bbox;

  const x1 =
    Number(bbox.x1 ?? 0);

  const y1 =
    Number(bbox.y1 ?? 0);

  const x2 =
    Number(bbox.x2 ?? 0);

  const y2 =
    Number(bbox.y2 ?? 0);

  const left =
    (x1 / imageWidth) * 100;

  const top =
    (y1 / imageHeight) * 100;

  const width =
    ((x2 - x1) /
      imageWidth) *
    100;

  const height =
    ((y2 - y1) /
      imageHeight) *
    100;

  const className =
    normalizeSubtype(
      detection.class_name
    );

  const borderColor =
    CLASS_COLORS[className] ||
    "#ffffff";

  return (
    <div
      className="detection-box"
      style={{
        left: `${left}%`,
        top: `${top}%`,
        width: `${width}%`,
        height: `${height}%`,
        borderColor,
      }}
    >
      <div
        className="detection-label"
        style={{
          borderColor,
          color: borderColor,
        }}
      >
        {className}{" "}
        {formatConfidence(
          detection.confidence
        )}
      </div>
    </div>
  );
}


/* =========================================================
   IMAGE DETECTION VIEWER
========================================================= */

function ImageDetectionViewer({
  result,
  localFile,
}) {
  const [showBoxes, setShowBoxes] =
    useState(true);

  const [localImageUrl, setLocalImageUrl] =
    useState(null);

  const [imageLoaded, setImageLoaded] =
    useState(false);

  const [actualImageSize, setActualImageSize] =
    useState({
      width: 0,
      height: 0,
    });

  const imageRef =
    useRef(null);

  /*
   * Backend image candidates.
   *
   * If backend provides an actual image,
   * use it first.
   */
  const backendImageSource =
    result?.image_preview ||
    result?.image_url ||
    result?.preview ||
    result?.image ||
    result?.annotated_image ||
    result?.annotated_image_url ||
    result?.result_image ||
    result?.result_image_url ||
    null;

  /*
   * Create a browser-local URL from
   * the original uploaded image.
   *
   * This is the important fallback that
   * prevents IMAGE PREVIEW UNAVAILABLE.
   */
  useEffect(() => {
    if (!localFile) {
      setLocalImageUrl(null);
      return;
    }

    const objectUrl =
      URL.createObjectURL(
        localFile
      );

    setLocalImageUrl(
      objectUrl
    );

    return () => {
      URL.revokeObjectURL(
        objectUrl
      );
    };
  }, [localFile]);

  /*
   * Backend URL may be a relative path.
   * Convert it into a usable URL.
   */
  const normalizeImageUrl = (
    source
  ) => {
    if (!source) {
      return null;
    }

    if (
      typeof source !== "string"
    ) {
      return null;
    }

    if (
      source.startsWith(
        "data:image/"
      ) ||
      source.startsWith(
        "blob:"
      ) ||
      source.startsWith(
        "http://"
      ) ||
      source.startsWith(
        "https://"
      )
    ) {
      return source;
    }

    if (
      source.startsWith("/")
    ) {
      return `http://127.0.0.1:8000${source}`;
    }

    return `http://127.0.0.1:8000/${source}`;
  };

  const remoteSource =
    normalizeImageUrl(
      backendImageSource
    );

  /*
   * Remote image first.
   * Uploaded image second.
   */
  const imageSource =
    remoteSource ||
    localImageUrl ||
    null;

  /*
   * If remote image fails, switch
   * immediately to uploaded image.
   */
  const [useLocalFallback, setUseLocalFallback] =
    useState(false);

  useEffect(() => {
    setUseLocalFallback(false);
    setImageLoaded(false);
  }, [
    remoteSource,
    localImageUrl,
    result?.filename,
  ]);

  const finalImageSource =
    useLocalFallback &&
    localImageUrl
      ? localImageUrl
      : imageSource;

  const imageWidth =
    actualImageSize.width ||
    Number(
      result?.image_size?.width
    ) ||
    Number(result?.width) ||
    640;

  const imageHeight =
    actualImageSize.height ||
    Number(
      result?.image_size?.height
    ) ||
    Number(result?.height) ||
    480;

  const detections =
    Array.isArray(
      result?.detections
    )
      ? result.detections
      : [];

  const handleImageLoad = (
    event
  ) => {
    const image =
      event.currentTarget;

    setImageLoaded(true);

    if (
      image.naturalWidth &&
      image.naturalHeight
    ) {
      setActualImageSize({
        width:
          image.naturalWidth,
        height:
          image.naturalHeight,
      });
    }
  };

  const handleImageError = () => {
    /*
     * If backend image cannot load,
     * use the actual uploaded file.
     */
    if (
      localImageUrl &&
      !useLocalFallback
    ) {
      setUseLocalFallback(true);
      setImageLoaded(false);
      return;
    }

    setImageLoaded(false);
  };

  return (
    <div className="image-viewer">
      {/* =====================================================
          TOOLBAR
      ===================================================== */}

      <div className="image-viewer-toolbar">
        <div>
          <span className="section-kicker">
            YOLO DETECTION
          </span>

          <h3>
            {result?.filename ||
              "Analyzed Image"}
          </h3>
        </div>

        <button
          type="button"
          className={`toggle-button ${
            showBoxes
              ? "toggle-active"
              : ""
          }`}
          onClick={() =>
            setShowBoxes(
              (previous) =>
                !previous
            )
          }
        >
          {showBoxes
            ? "HIDE BOXES"
            : "SHOW BOXES"}
        </button>
      </div>

      {/* =====================================================
          IMAGE
      ===================================================== */}

      {finalImageSource ? (
        <div className="image-stage">
          <img
            ref={imageRef}
            src={finalImageSource}
            alt={
              result?.filename ||
              "Blood smear"
            }
            className="analysis-image"
            onLoad={
              handleImageLoad
            }
            onError={
              handleImageError
            }
          />

          {showBoxes &&
            imageLoaded &&
            detections.map(
              (
                detection,
                index
              ) => (
                <DetectionOverlay
                  key={`${result?.filename}-${index}`}
                  detection={
                    detection
                  }
                  imageWidth={
                    imageWidth
                  }
                  imageHeight={
                    imageHeight
                  }
                />
              )
            )}
        </div>
      ) : (
        <div className="image-placeholder">
          <span>
            IMAGE PREVIEW UNAVAILABLE
          </span>
        </div>
      )}
    </div>
  );
}


/* =========================================================
   WBC RESULT CARD
========================================================= */

function WBCResultCard({
  item,
  index,
}) {
  const classification =
    item?.classification || {};

  const finalSubtype =
    normalizeSubtype(
      classification.final_decision ||
        classification.class_name ||
        classification.subtype
    );

  const decisionMethod =
    classification.decision_method ||
    classification.method ||
    "";

  const foldPredictions =
    Array.isArray(
      classification.fold_predictions
    )
      ? classification.fold_predictions
      : [];

  const probabilities =
    classification.probabilities ||
    {};

  const confidence =
    classification.confidence ??
    classification.ensemble_confidence ??
    0;

  const foldAgreement =
    classification.fold_agreement ??
    0;

  const votes =
    classification.votes ??
    classification.vote_count ??
    0;

  const voteTotal =
    classification.vote_total ??
    3;

  const reliability =
    classification.reliability ||
    "";

  return (
    <div className="wbc-result-card">
      {/* =====================================================
          WBC CARD HEADER
      ===================================================== */}

      <div className="wbc-card-top">
        <div>
          <span className="section-kicker">
            WBC {index + 1}
          </span>

          <h3>
            Subtype Classification
          </h3>
        </div>

        <div className="wbc-detection-confidence">
          YOLO{" "}
          {formatConfidence(
            item?.yolo_confidence
          )}
        </div>
      </div>

      {/* =====================================================
          FINAL RESULT
      ===================================================== */}

      <div className="final-result-grid">
        <div className="final-result-main">
          <span className="final-result-label">
            FINAL RESULT
          </span>

          <div
            className={`final-subtype ${getSubtypeClass(
              finalSubtype
            )}`}
          >
            {finalSubtype}
          </div>

          <div className="final-confidence">
            {formatConfidence(
              confidence
            )}
          </div>

          <span className="final-confidence-text">
            FINAL CONFIDENCE
          </span>
        </div>

        {/* =================================================
            DECISION PANEL
        ================================================= */}

        <div className="decision-panel">
          <span className="decision-label">
            DECISION METHOD
          </span>

          <div className="decision-method">
            {getDecisionMethodLabel(
              decisionMethod
            )}
          </div>

          <p className="decision-description">
            {getDecisionDescription(
              decisionMethod
            )}
          </p>

          <div className="decision-meta">
            <div>
              <span>
                AGREEMENT
              </span>

              <strong>
                {formatPercent(
                  foldAgreement
                )}
              </strong>
            </div>

            <div>
              <span>
                VOTES
              </span>

              <strong>
                {votes}/{voteTotal}
              </strong>
            </div>
          </div>
        </div>
      </div>

      {/* =====================================================
          FOLD PREDICTIONS
      ===================================================== */}

      {foldPredictions.length >
        0 && (
        <div className="fold-section">
          <div className="section-heading">
            <div>
              <span className="section-kicker">
                ENSEMBLE ANALYSIS
              </span>

              <h3>
                3-Fold Predictions
              </h3>
            </div>

            <span className="fold-count">
              {foldPredictions.length}{" "}
              FOLDS
            </span>
          </div>

          <div className="fold-grid">
            {foldPredictions.map(
              (
                fold,
                foldIndex
              ) => (
                <FoldCard
                  key={`fold-${foldIndex}`}
                  fold={fold}
                  index={
                    foldIndex
                  }
                  finalSubtype={
                    finalSubtype
                  }
                />
              )
            )}
          </div>
        </div>
      )}

      {/* =====================================================
          PROBABILITIES
      ===================================================== */}

      <ProbabilityPanel
        probabilities={
          probabilities
        }
        finalSubtype={
          finalSubtype
        }
      />

      {/* =====================================================
          CLASSIFIER STATUS
      ===================================================== */}

      {reliability && (
        <div className="reliability-row">
          <span>
            CLASSIFIER STATUS
          </span>

          <strong
            className={
              finalSubtype ===
              "UNCERTAIN"
                ? "status-warning"
                : "status-success"
            }
          >
            {finalSubtype ===
            "UNCERTAIN"
              ? reliability.toUpperCase()
              : "COMPLETED"}
          </strong>
        </div>
      )}

      {/* =====================================================
          CLASSIFIER INPUT
          KEEP THIS VISIBLE
      ===================================================== */}

      {item?.crop_image && (
        <div className="crop-preview">
          <div className="crop-header">
            <span className="section-kicker">
              CLASSIFIER INPUT
            </span>

            <span>
              WBC CROP
            </span>
          </div>

          <img
            src={item.crop_image}
            alt={`WBC ${
              index + 1
            } crop`}
          />
        </div>
      )}
    </div>
  );
}


/* =========================================================
   MAIN ANALYSIS COMPONENT
========================================================= */

export default function Analysis() {
  const [
    selectedFiles,
    setSelectedFiles,
  ] = useState([]);

  const [
    results,
    setResults,
  ] = useState(null);

  const [
    loading,
    setLoading,
  ] = useState(false);

  const [
    error,
    setError,
  ] = useState("");

  const [
    dragActive,
    setDragActive,
  ] = useState(false);

  const [
    activeImage,
    setActiveImage,
  ] = useState(0);

  const fileInputRef =
    useRef(null);

  const backgroundVideoRef =
    useRef(null);


  /* =======================================================
     BACKGROUND VIDEO
  ======================================================= */

  useEffect(() => {
    const video =
      backgroundVideoRef.current;

    if (!video) {
      return;
    }

    video.muted = true;
    video.defaultMuted = true;
    video.loop = true;
    video.playsInline = true;

    const playVideo = () => {
      video
        .play()
        .catch((err) => {
          console.warn(
            "Background video autoplay attempt failed:",
            err
          );
        });
    };

    playVideo();

    video.addEventListener(
      "loadeddata",
      playVideo
    );

    video.addEventListener(
      "canplay",
      playVideo
    );

    video.addEventListener(
      "loadedmetadata",
      playVideo
    );

    return () => {
      video.removeEventListener(
        "loadeddata",
        playVideo
      );

      video.removeEventListener(
        "canplay",
        playVideo
      );

      video.removeEventListener(
        "loadedmetadata",
        playVideo
      );
    };
  }, []);


  /* =======================================================
     FILE HANDLING
  ======================================================= */

  const addFiles = (files) => {
    const incoming =
      Array.from(files || []);

    const imageFiles =
      incoming.filter(
        (file) =>
          file.type.startsWith(
            "image/"
          )
      );

    if (!imageFiles.length) {
      setError(
        "Please select valid image files."
      );
      return;
    }

    setError("");

    setSelectedFiles(
      (previous) => {
        const combined = [
          ...previous,
          ...imageFiles,
        ];

        const unique =
          combined.filter(
            (
              file,
              index,
              array
            ) =>
              index ===
              array.findIndex(
                (other) =>
                  other.name ===
                    file.name &&
                  other.size ===
                    file.size &&
                  other.lastModified ===
                    file.lastModified
              )
          );

        return unique.slice(
          0,
          50
        );
      }
    );

    setResults(null);
    setActiveImage(0);
  };


  const removeFile = (
    index
  ) => {
    setSelectedFiles(
      (previous) =>
        previous.filter(
          (
            _,
            fileIndex
          ) =>
            fileIndex !==
            index
        )
    );
  };


  const clearFiles = () => {
    setSelectedFiles([]);
    setResults(null);
    setError("");
    setActiveImage(0);

    if (fileInputRef.current) {
      fileInputRef.current.value =
        "";
    }
  };


  /* =======================================================
     DRAG & DROP
  ======================================================= */

  const handleDragOver = (
    event
  ) => {
    event.preventDefault();
    event.stopPropagation();

    setDragActive(true);
  };


  const handleDragLeave = (
    event
  ) => {
    event.preventDefault();
    event.stopPropagation();

    setDragActive(false);
  };


  const handleDrop = (
    event
  ) => {
    event.preventDefault();
    event.stopPropagation();

    setDragActive(false);

    addFiles(
      event.dataTransfer.files
    );
  };


  /* =======================================================
     ANALYZE
  ======================================================= */

  const analyzeImages =
    async () => {
      if (!selectedFiles.length) {
        setError(
          "Please upload at least one image."
        );
        return;
      }

      setLoading(true);
      setError("");
      setResults(null);
      setActiveImage(0);

      const formData =
        new FormData();

      selectedFiles.forEach(
        (file) => {
          formData.append(
            "images",
            file
          );
        }
      );

      try {
        const response =
          await fetch(
            API_URL,
            {
              method: "POST",
              body: formData,
            }
          );

        const data =
          await response.json();

        if (!response.ok) {
          throw new Error(
            data?.detail ||
              data?.message ||
              "Analysis failed."
          );
        }

        if (
          data?.status &&
          data.status !==
            "success"
        ) {
          throw new Error(
            data?.message ||
              "Analysis failed."
          );
        }

        setResults(data);
      } catch (err) {
        console.error(
          "Analysis error:",
          err
        );

        setError(
          err?.message ||
            "Unable to connect to the analysis server."
        );
      } finally {
        setLoading(false);
      }
    };


  /* =======================================================
     SUMMARY DATA
  ======================================================= */

  const totalCounts =
    results?.total_counts || {
      WBC: 0,
      RBC: 0,
      Platelets: 0,
    };

  const imageResults =
    Array.isArray(
      results?.results
    )
      ? results.results
      : [];

  const globalWbcSummary =
    results?.wbc_subtype_analysis ||
    {};

  const totalWBC =
    Number(
      totalCounts.WBC || 0
    );

  const classifiedWBC =
    Number(
      globalWbcSummary.successfully_classified ||
        0
    );

  const averageConfidence =
    globalWbcSummary.average_confidence ||
    0;


  /* =======================================================
     FIND ORIGINAL UPLOADED FILE
  ======================================================= */

  const findLocalFile = (
    result
  ) => {
    if (
      !result ||
      !selectedFiles.length
    ) {
      return null;
    }

    const resultName =
      result.filename ||
      result.file_name ||
      result.name;

    if (!resultName) {
      return null;
    }

    return (
      selectedFiles.find(
        (file) =>
          file.name ===
          resultName
      ) || null
    );
  };


  /* =======================================================
     RENDER
  ======================================================= */

  return (
    <div className="analysis-page">

      {/* =================================================
          BACKGROUND VIDEO
      ================================================= */}

      <video
        ref={backgroundVideoRef}
        className="analysis-background-video"
        autoPlay
        muted
        defaultMuted
        loop
        playsInline
        preload="auto"
        aria-hidden="true"
      >
        <source
          src={bloodBackground}
          type="video/mp4"
        />
      </video>

      <div className="analysis-video-overlay" />


      {/* =================================================
          HEADER
      ================================================= */}

      <header className="analysis-header">
        <div className="analysis-brand">

          <div className="brand-mark">
            AI
          </div>

          <div>
            <span className="brand-small">
              INTELLIGENT HEMATOLOGY
            </span>

            <h1>
              Blood Cell Analysis
            </h1>
          </div>

        </div>

        <div className="analysis-status">
          <span className="status-dot" />

          <span>
            {loading
              ? "ANALYZING"
              : "SYSTEM READY"}
          </span>
        </div>
      </header>


      {/* =================================================
          MAIN CONTENT
      ================================================= */}

      <main className="analysis-content">

        {/* =================================================
            UPLOAD PANEL
        ================================================= */}

        {!results && (
          <section className="upload-section">

            <div
              className={`upload-card ${
                dragActive
                  ? "drag-active"
                  : ""
              }`}
              onDragOver={
                handleDragOver
              }
              onDragLeave={
                handleDragLeave
              }
              onDrop={
                handleDrop
              }
              onClick={() =>
                fileInputRef.current?.click()
              }
            >

              <input
                ref={
                  fileInputRef
                }
                type="file"
                accept="image/*"
                multiple
                hidden
                onChange={(
                  event
                ) =>
                  addFiles(
                    event.target
                      .files
                  )
                }
              />

              <div className="upload-icon">
                +
              </div>

              <span className="upload-kicker">
                BLOOD SMEAR ANALYSIS
              </span>

              <h2>
                Upload Microscopy Images
              </h2>

              <p>
                Drag and drop your
                blood-smear images
                here, or click to
                browse.
              </p>

              <span className="upload-limit">
                JPG · JPEG · PNG · BMP · TIFF · WEBP
                &nbsp; / &nbsp;
                MAX 50 IMAGES
              </span>

            </div>


            {/* =================================================
                SELECTED FILES
            ================================================= */}

            {selectedFiles.length >
              0 && (
              <div className="selected-files">

                <div className="selected-files-header">

                  <div>
                    <span className="section-kicker">
                      SELECTED FILES
                    </span>

                    <h3>
                      {
                        selectedFiles.length
                      }{" "}
                      image
                      {selectedFiles.length !==
                      1
                        ? "s"
                        : ""}
                    </h3>
                  </div>

                  <button
                    type="button"
                    className="text-button"
                    onClick={
                      clearFiles
                    }
                  >
                    CLEAR ALL
                  </button>

                </div>


                <div className="file-list">

                  {selectedFiles.map(
                    (
                      file,
                      index
                    ) => (
                      <div
                        className="file-item"
                        key={`${file.name}-${index}`}
                        onClick={(
                          event
                        ) =>
                          event.stopPropagation()
                        }
                      >

                        <div className="file-number">
                          {String(
                            index +
                              1
                          ).padStart(
                            2,
                            "0"
                          )}
                        </div>

                        <div className="file-info">
                          <strong>
                            {
                              file.name
                            }
                          </strong>

                          <span>
                            {(
                              file.size /
                              1024 /
                              1024
                            ).toFixed(
                              2
                            )}{" "}
                            MB
                          </span>
                        </div>

                        <button
                          type="button"
                          className="remove-file"
                          onClick={() =>
                            removeFile(
                              index
                            )
                          }
                        >
                          ×
                        </button>

                      </div>
                    )
                  )}

                </div>


                <button
                  type="button"
                  className="analyze-button"
                  onClick={(
                    event
                  ) => {
                    event.stopPropagation();
                    analyzeImages();
                  }}
                  disabled={
                    loading
                  }
                >
                  {loading
                    ? "ANALYZING..."
                    : "START ANALYSIS"}

                  <span>
                    →
                  </span>
                </button>

              </div>
            )}

          </section>
        )}


        {/* =================================================
            ERROR
        ================================================= */}

        {error && (
          <div className="error-message">

            <span>!</span>

            <div>
              <strong>
                ANALYSIS ERROR
              </strong>

              <p>
                {error}
              </p>
            </div>

          </div>
        )}


        {/* =================================================
            RESULTS
        ================================================= */}

        {results && (
          <section className="results-section">

            {/* =================================================
                RESULTS HEADER
            ================================================= */}

            <div className="results-header">

              <div>
                <span className="section-kicker">
                  ANALYSIS COMPLETE
                </span>

                <h2>
                  Blood Cell Analysis Results
                </h2>

                <p>
                  {results.image_count ||
                    imageResults.length}{" "}
                  image
                  {(results.image_count ||
                    imageResults.length) !==
                  1
                    ? "s"
                    : ""}{" "}
                  processed
                  successfully.
                </p>
              </div>

              <button
                type="button"
                className="new-analysis-button"
                onClick={
                  clearFiles
                }
              >
                + NEW ANALYSIS
              </button>

            </div>


            {/* =================================================
                COUNT SUMMARY
            ================================================= */}

            <div className="count-grid">

              <CountCard
                title="WHITE BLOOD CELLS"
                value={
                  totalCounts.WBC
                }
                type="WBC"
              />

              <CountCard
                title="RED BLOOD CELLS"
                value={
                  totalCounts.RBC
                }
                type="RBC"
              />

              <CountCard
                title="PLATELETS"
                value={
                  totalCounts.Platelets
                }
                type="Platelets"
              />

              <div className="count-card count-analysis">

                <div className="count-card-top">
                  <span className="count-card-label">
                    WBC CLASSIFIED
                  </span>
                </div>

                <div className="count-card-value">

                  {
                    classifiedWBC
                  }

                  <span className="count-card-total">
                    / {totalWBC}
                  </span>

                </div>

              </div>

            </div>


            {/* =================================================
                GLOBAL CLASSIFIER SUMMARY
            ================================================= */}

            {totalWBC > 0 && (
              <div className="global-summary">

                <div className="global-summary-main">

                  <span className="section-kicker">
                    CONVNEXT ENSEMBLE
                  </span>

                  <h3>
                    WBC Subtype Classification
                  </h3>

                  <p>
                    Each detected
                    WBC is evaluated
                    using three
                    independent
                    model folds.
                  </p>

                </div>


                {/* =================================================
                    AVERAGE CONFIDENCE
                    KEEPING EXISTING DATA
                ================================================= */}

                <div className="global-summary-stat">
                  <span>
                    AVERAGE CONFIDENCE
                  </span>

                  <strong>
                    {formatConfidence(
                      averageConfidence
                    )}
                  </strong>
                </div>


                <div className="global-summary-stat">

                  <span>
                    CLASSIFIED CELLS
                  </span>

                  <strong>
                    {classifiedWBC}/
                    {totalWBC}
                  </strong>

                </div>

              </div>
            )}


            {/* =================================================
                IMAGE SELECTOR
            ================================================= */}

            {imageResults.length >
              1 && (
              <div className="image-tabs">

                {imageResults.map(
                  (
                    result,
                    index
                  ) => (
                    <button
                      type="button"
                      key={`tab-${index}`}
                      className={
                        activeImage ===
                        index
                          ? "image-tab active"
                          : "image-tab"
                      }
                      onClick={() =>
                        setActiveImage(
                          index
                        )
                      }
                    >

                      <span>
                        {String(
                          index +
                            1
                        ).padStart(
                          2,
                          "0"
                        )}
                      </span>

                      {
                        result.filename
                      }

                    </button>
                  )
                )}

              </div>
            )}


            {/* =================================================
                ACTIVE IMAGE WITH YOLO BOXES
            ================================================= */}

            {imageResults.length >
              0 && (
              <div className="result-image-section">

                <ImageDetectionViewer
                  result={
                    imageResults[
                      Math.min(
                        activeImage,
                        imageResults.length -
                          1
                      )
                    ]
                  }
                  localFile={findLocalFile(
                    imageResults[
                      Math.min(
                        activeImage,
                        imageResults.length -
                          1
                      )
                    ]
                  )}
                />

              </div>
            )}


            {/* =================================================
                WBC CLASSIFICATION RESULTS
            ================================================= */}

            {imageResults.map(
              (
                result,
                imageIndex
              ) => {

                const isActive =
                  imageResults.length ===
                    1 ||
                  activeImage ===
                    imageIndex;

                if (!isActive) {
                  return null;
                }

                const classifications =
                  Array.isArray(
                    result.wbc_classifications
                  )
                    ? result.wbc_classifications
                    : [];

                return (
                  <div
                    className="wbc-results-section"
                    key={`wbc-results-${imageIndex}`}
                  >

                    <div className="section-main-heading">

                      <div>
                        <span className="section-kicker">
                          WBC SUBTYPE RESULTS
                        </span>

                        <h2>
                          Detected White Blood Cells
                        </h2>
                      </div>

                      <div className="section-result-count">
                        {
                          classifications.length
                        }{" "}
                        DETECTED
                      </div>

                    </div>


                    {classifications.length >
                    0 ? (
                      <div className="wbc-results-grid">

                        {classifications.map(
                          (
                            item,
                            index
                          ) => (
                            <WBCResultCard
                              key={`wbc-${imageIndex}-${index}`}
                              item={
                                item
                              }
                              index={
                                index
                              }
                            />
                          )
                        )}

                      </div>
                    ) : (
                      <div className="no-wbc-message">
                        <span>
                          NO WBC CELLS DETECTED
                        </span>
                      </div>
                    )}

                  </div>
                );
              }
            )}


            {/* =================================================
                FOOTER
            ================================================= */}

            <div className="results-footer">

              <div>

                <span className="section-kicker">
                  INFERENCE PIPELINE
                </span>

                <p>
                  YOLOv11 BCCD
                  detection →
                  WBC extraction →
                  3-fold ConvNeXt
                  classification →
                  majority vote or
                  highest probability.
                </p>

              </div>

              <button
                type="button"
                className="new-analysis-button footer-button"
                onClick={
                  clearFiles
                }
              >
                RUN ANOTHER ANALYSIS →
              </button>

            </div>

          </section>
        )}

      </main>
    </div>
  );
}