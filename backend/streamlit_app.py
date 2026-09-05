from __future__ import annotations

import base64
import gc
import sys
import tempfile

from pathlib import Path
from typing import Any, Dict

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import streamlit.components.v1 as components


# ============================================================
# MARKDOWN / HTML INDENTATION FIX
# ============================================================
# Python triple-quoted HTML strings are indented for readability.
# Streamlit's Markdown parser can interpret 4+ leading spaces as a
# code block, which makes the raw <div> / <style> tags appear on
# the webpage. Dedent all markdown content before rendering it.


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent

BACKEND_DIR = ROOT_DIR

ASSETS_DIR = ROOT_DIR / "assets"

HOME_VIDEO = ASSETS_DIR / "homepage.mp4"


# ============================================================
# BACKEND IMPORT PATH
# ============================================================

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Blood Cell Intelligence",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# BACKEND MODELS
# ============================================================

try:

    from app.services import detector
    from app.services import classifier

except Exception as exc:

    st.error(
        "Failed to load the backend model modules."
    )

    st.code(str(exc))

    st.stop()


# ============================================================
# CONSTANTS
# ============================================================

MAX_FILE_SIZE_MB = 10

# Minimum confidence required for a WBC detection before classification.
MIN_WBC_VALIDATION_CONFIDENCE = 0.50

CLASS_COLORS = {
    "WBC": (255, 80, 120),
    "RBC": (255, 70, 70),
    "Platelets": (100, 220, 180),
}


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:

    st.session_state.page = "home"


if "analysis_results" not in st.session_state:

    st.session_state.analysis_results = None

# ============================================================
# URL / PAGE NAVIGATION
# ============================================================

query_page = st.query_params.get("page")

if query_page == "analysis":
    st.session_state.page = "analysis"

elif query_page == "home":
    st.session_state.page = "home"
# ============================================================
# QUERY PARAMETER NAVIGATION
# ============================================================

try:

    query_page = st.query_params.get(
        "page",
        None,
    )

    if query_page == "analysis":

        st.session_state.page = "analysis"

    elif query_page == "home":

        st.session_state.page = "home"

except Exception:

    pass


# ============================================================
# GLOBAL CSS
# ============================================================

st.html(
    """
    <style>

    /* --------------------------------------------------------
       GLOBAL
    -------------------------------------------------------- */

    html,
    body,
    [data-testid="stAppViewContainer"] {

        background:
            radial-gradient(
                circle at 50% 20%,
                rgba(70, 40, 100, 0.20),
                transparent 40%
            ),
            #05060b !important;

        color: #ffffff !important;

    }


    [data-testid="stHeader"] {

        background: transparent !important;

    }


    [data-testid="stToolbar"] {

        visibility: hidden;

    }


    footer {

        visibility: hidden;

    }


    /* --------------------------------------------------------
       REMOVE STREAMLIT PADDING
    -------------------------------------------------------- */

    .block-container {

        padding-top: 0 !important;
        padding-right: 0 !important;
        padding-bottom: 0 !important;
        padding-left: 0 !important;
        max-width: none !important;

    }


    /* --------------------------------------------------------
       BUTTONS
    -------------------------------------------------------- */

    .stButton > button {

        border-radius: 12px !important;

        border: 1px solid
            rgba(255, 255, 255, 0.14) !important;

        background:
            linear-gradient(
                135deg,
                rgba(255, 70, 110, 0.90),
                rgba(150, 60, 180, 0.90)
            ) !important;

        color: white !important;

        font-weight: 700 !important;

        min-height: 48px !important;

        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease !important;

    }


    .stButton > button:hover {

        transform: translateY(-2px);

        box-shadow:
            0 12px 35px
            rgba(255, 60, 120, 0.25) !important;

    }


    /* --------------------------------------------------------
       FILE UPLOADER
    -------------------------------------------------------- */

    [data-testid="stFileUploader"] {

        background:
            rgba(255, 255, 255, 0.035);

        border:
            1px solid
            rgba(255, 255, 255, 0.10);

        border-radius: 20px;

        padding: 18px;

    }


    /* --------------------------------------------------------
       METRICS
    -------------------------------------------------------- */

    [data-testid="stMetric"] {

        background:
            rgba(255, 255, 255, 0.035);

        border:
            1px solid
            rgba(255, 255, 255, 0.10);

        border-radius: 18px;

        padding: 20px;

        backdrop-filter: blur(15px);

    }


    /* --------------------------------------------------------
       DIVIDERS
    -------------------------------------------------------- */

    hr {

        border-color:
            rgba(255, 255, 255, 0.10) !important;

    }

    </style>
    """
)


# ============================================================
# VIDEO → BASE64
# ============================================================

@st.cache_data(show_spinner=False)
def load_video_base64(
    video_path: str,
) -> str:

    path = Path(video_path)

    if not path.exists():

        return ""

    data = path.read_bytes()

    encoded = base64.b64encode(
        data
    ).decode("utf-8")

    return encoded


# ============================================================
# HOME PAGE
# ============================================================

def render_home():
    """Render the cinematic full-screen landing page.

    The homepage video is displayed as a fixed full-window background.
    A real Streamlit button is positioned over the centre of the video.
    Using a real Streamlit button is intentional: it guarantees a Streamlit
    rerun and therefore reliable navigation to the analysis page.
    """

    if not HOME_VIDEO.exists():
        st.error("Homepage video not found.")
        st.code(str(HOME_VIDEO))
        st.info("Make sure assets/homepage.mp4 exists.")
        return

    video_base64 = load_video_base64(str(HOME_VIDEO))

    if not video_base64:
        st.error("Unable to load homepage video.")
        return

    # Home-only CSS. The Streamlit button is the only interactive control
    # on the home page, so it can safely be positioned over the WBC.
    st.html(
        """
        <style>
            html, body {
                overflow: hidden !important;
                margin: 0 !important;
                padding: 0 !important;
            }

            [data-testid="stHeader"],
            [data-testid="stToolbar"],
            footer {
                display: none !important;
            }

            .home-screen {
                position: fixed;
                inset: 0;
                width: 100vw;
                height: 100vh;
                overflow: hidden;
                background: #05060b;
                z-index: 999999;
                font-family: Inter, -apple-system, BlinkMacSystemFont,
                    "Segoe UI", sans-serif;
            }

            .home-screen video {
                position: absolute;
                inset: 0;
                width: 100%;
                height: 100%;
                display: block;
                object-fit: cover;
                object-position: center center;
                background: #05060b;
            }

            .home-overlay {
                position: absolute;
                inset: 0;
                z-index: 2;
                pointer-events: none;
                background: linear-gradient(
                    to bottom,
                    rgba(4, 6, 18, 0.14) 0%,
                    rgba(4, 6, 18, 0.00) 32%,
                    rgba(4, 6, 18, 0.03) 70%,
                    rgba(4, 6, 18, 0.22) 100%
                );
            }

            .home-brand {
                position: absolute;
                top: clamp(24px, 5vh, 52px);
                left: clamp(26px, 5vw, 72px);
                z-index: 5;
                color: rgba(255, 255, 255, 0.96);
                font-size: clamp(11px, 0.85vw, 15px);
                font-weight: 800;
                letter-spacing: 4px;
                text-transform: uppercase;
                text-shadow: 0 2px 20px rgba(0, 0, 0, 0.35);
                pointer-events: none;
            }

            .home-brand::before {
                content: "";
                display: inline-block;
                width: 7px;
                height: 7px;
                margin-right: 10px;
                border-radius: 50%;
                background: #ffffff;
                vertical-align: 2px;
                box-shadow: 0 0 18px rgba(255, 255, 255, 0.8);
            }

            .home-instruction {
                position: absolute;
                left: 50%;
                bottom: clamp(24px, 5vh, 54px);
                z-index: 5;
                transform: translateX(-50%);
                padding: 12px 22px;
                border: 1px solid rgba(255, 255, 255, 0.16);
                border-radius: 999px;
                background: rgba(8, 8, 18, 0.42);
                color: rgba(255, 255, 255, 0.88);
                font-size: clamp(10px, 0.75vw, 13px);
                font-weight: 800;
                letter-spacing: 2.2px;
                text-transform: uppercase;
                white-space: nowrap;
                backdrop-filter: blur(14px);
                -webkit-backdrop-filter: blur(14px);
                pointer-events: none;
                text-shadow: 0 2px 14px rgba(0, 0, 0, 0.4);
            }

            /* The only home-page control. It sits exactly over the
               central WBC and is therefore the clickable hotspot. */
            .home-start-button {
                position: fixed !important;
                left: 50% !important;
                top: 50% !important;
                transform: translate(-50%, -50%) !important;
                width: clamp(190px, 16vw, 250px) !important;
                z-index: 1000001 !important;
                margin: 0 !important;
                padding: 0 !important;
            }

            .home-start-button > button {
                width: 100% !important;
                height: 54px !important;
                border-radius: 999px !important;
                border: 1px solid rgba(255, 255, 255, 0.24) !important;
                background: rgba(38, 12, 70, 0.88) !important;
                color: #ffffff !important;
                font-size: 12px !important;
                font-weight: 900 !important;
                letter-spacing: 2.4px !important;
                text-transform: uppercase !important;
                box-shadow: 0 0 45px rgba(150, 70, 255, 0.22),
                    0 15px 45px rgba(0, 0, 0, 0.28) !important;
                backdrop-filter: blur(14px) !important;
                -webkit-backdrop-filter: blur(14px) !important;
                cursor: pointer !important;
                transition: all 0.22s ease !important;
            }

            .home-start-button > button:hover {
                transform: scale(1.05) !important;
                border-color: rgba(255, 255, 255, 0.50) !important;
                box-shadow: 0 0 65px rgba(190, 90, 255, 0.38),
                    0 18px 55px rgba(0, 0, 0, 0.34) !important;
            }

            .home-start-button > button:active {
                transform: scale(0.97) !important;
            }

            @media (max-width: 700px) {
                .home-brand {
                    letter-spacing: 2.5px;
                }

                .home-instruction {
                    letter-spacing: 1.1px;
                    padding: 10px 15px;
                }

                .home-start-button {
                    width: 190px !important;
                }
            }
        </style>
        """
    )

    video_html = f"""
        <div class="home-screen">
            <video autoplay muted loop playsinline preload="auto" aria-hidden="true">
                <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
            </video>

            <div class="home-overlay"></div>

            <div class="home-brand">Blood Cell Intelligence</div>

            <div class="home-instruction">
                Click the central white blood cell to begin
            </div>
        </div>
    """

    st.html(video_html)

    # IMPORTANT: This is a real Streamlit widget, not HTML/JavaScript.
    # Clicking it causes Streamlit to rerun the script.
    button_placeholder = st.empty()
    with button_placeholder.container():
        if st.button(
    "START ANALYSIS",
    key="home_start_analysis",
    type="primary",
    use_container_width=True,
):
    st.session_state.page = "analysis"
    st.session_state.analysis_results = None

    st.query_params["page"] = "analysis"

    st.rerun()

    # Give the Streamlit button wrapper the class used by the home CSS.
    st.html(
        """
        <style>
            /* The home page contains exactly one Streamlit button. */
            div[data-testid="stButton"] {
                position: fixed !important;
                left: 50% !important;
                top: 50% !important;
                transform: translate(-50%, -50%) !important;
                width: clamp(190px, 16vw, 250px) !important;
                z-index: 1000001 !important;
                margin: 0 !important;
            }

            div[data-testid="stButton"] > button {
                width: 100% !important;
            }
        </style>
        """
    )


# ============================================================
# DRAW DETECTIONS
# ============================================================

def draw_detections(
    image: Image.Image,
    detections: list[Dict[str, Any]],
) -> Image.Image:

    output = (
        image
        .copy()
        .convert("RGB")
    )

    draw = ImageDraw.Draw(
        output
    )

    try:

        font = ImageFont.load_default()

    except Exception:

        font = None


    for detection in detections:

        class_name = detection.get(
            "class_name",
            "Unknown",
        )

        confidence = float(
            detection.get(
                "confidence",
                0.0,
            )
        )

        bbox = detection.get(
            "bbox",
            {},
        )

        x1 = int(
            float(
                bbox.get("x1", 0)
            )
        )

        y1 = int(
            float(
                bbox.get("y1", 0)
            )
        )

        x2 = int(
            float(
                bbox.get("x2", 0)
            )
        )

        y2 = int(
            float(
                bbox.get("y2", 0)
            )
        )

        color = CLASS_COLORS.get(
            class_name,
            (255, 255, 255),
        )


        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        draw.rectangle(
            [
                x1,
                y1,
                x2,
                y2,
            ],
            outline=color,
            width=3,
        )


        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        label = (
            f"{class_name} "
            f"{confidence * 100:.1f}%"
        )


        if font is not None:

            try:

                bbox_text = draw.textbbox(
                    (0, 0),
                    label,
                    font=font,
                )

                text_width = (
                    bbox_text[2]
                    - bbox_text[0]
                )

                text_height = (
                    bbox_text[3]
                    - bbox_text[1]
                )

            except Exception:

                text_width = (
                    len(label) * 7
                )

                text_height = 12

        else:

            text_width = (
                len(label) * 7
            )

            text_height = 12


        text_x = x1

        text_y = max(
            0,
            y1 - text_height - 6,
        )


        draw.rectangle(
            [
                text_x,
                text_y,
                text_x
                + text_width
                + 8,
                text_y
                + text_height
                + 6,
            ],
            fill=color,
        )


        draw.text(
            (
                text_x + 4,
                text_y + 3,
            ),
            label,
            fill=(0, 0, 0),
            font=font,
        )


    return output


# ============================================================
# CLASSIFY WBCs
# ============================================================

def crop_wbc_from_image(
    image: Image.Image,
    bbox: Dict[str, Any],
    padding: float = 0.15,
) -> Image.Image:
    """Safely crop one WBC directly from a PIL image."""
    source = image.convert("RGB")

    width, height = source.size

    x1 = float(bbox.get("x1", 0))
    y1 = float(bbox.get("y1", 0))
    x2 = float(bbox.get("x2", 0))
    y2 = float(bbox.get("y2", 0))

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid WBC bounding box.")

    box_width = x2 - x1
    box_height = y2 - y1

    x1 -= box_width * padding
    y1 -= box_height * padding
    x2 += box_width * padding
    y2 += box_height * padding

    x1 = max(0, min(int(x1), width))
    y1 = max(0, min(int(y1), height))
    x2 = max(0, min(int(x2), width))
    y2 = max(0, min(int(y2), height))

    if x2 <= x1 or y2 <= y1:
        raise ValueError("WBC crop has invalid dimensions.")

    return source.crop((x1, y1, x2, y2))


def classify_detected_wbcs(
    wbc_crops: list[Dict[str, Any]],
) -> list[Dict[str, Any]]:

    results = []


    for item in wbc_crops:

        crop = item.get(
            "crop"
        )

        if crop is None:

            continue


        try:

            result = (
                classifier
                .classify_wbc_image(
                    crop
                )
            )


            results.append(
                {
                    "wbc_index":
                        item.get(
                            "wbc_index",
                            len(results) + 1,
                        ),

                    "detection_confidence":
                        item.get(
                            "confidence",
                            0.0,
                        ),

                    "bbox":
                        item.get(
                            "bbox",
                            {},
                        ),

                    "classification":
                        result,

                    "crop":
                        crop,
                }
            )


        except Exception as exc:

            results.append(
                {
                    "wbc_index":
                        item.get(
                            "wbc_index",
                            len(results) + 1,
                        ),

                    "detection_confidence":
                        item.get(
                            "confidence",
                            0.0,
                        ),

                    "bbox":
                        item.get(
                            "bbox",
                            {},
                        ),

                    "classification":
                        {
                            "error":
                                str(exc)
                        },

                    "crop":
                        crop,
                }
            )


    return results


# ============================================================
# ANALYSIS PAGE
# ============================================================

def render_analysis():

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    header_col1, header_col2 = st.columns(
        [5, 1]
    )


    with header_col1:

        st.html(
            """
            <div style="padding-top:15px;">

                <div
                    style="
                        color:#ff6b91;
                        font-size:12px;
                        letter-spacing:3px;
                        font-weight:700;
                    "
                >
                    ANALYSIS WORKSPACE
                </div>

                <div
                    style="
                        font-size:42px;
                        font-weight:800;
                        margin-top:5px;
                    "
                >
                    Blood Cell Analysis
                </div>

                <div
                    style="
                        color:#8d8e9b;
                        font-size:16px;
                        margin-top:5px;
                    "
                >
                    AI-assisted blood smear
                    detection and WBC classification
                </div>

            </div>
            """
        )


    with header_col2:

        st.write("")
      if st.button(
    "← Home",
    key="analysis_home_button",
    use_container_width=True,
):
    st.session_state.page = "home"
    st.session_state.analysis_results = None

    st.query_params["page"] = "home"

    st.rerun()
        


    st.divider()


    # ========================================================
    # SIDEBAR
    # ========================================================

    with st.sidebar:

        st.markdown(
            """
            <div
                style="
                    font-size:22px;
                    font-weight:800;
                    margin-bottom:20px;
                "
            >
                🩸 Blood Cell
            </div>
            """,
            unsafe_allow_html=True,
        )


        st.html(
            """
            <div
                style="
                    color:#ff6b91;
                    font-size:11px;
                    letter-spacing:2px;
                    font-weight:700;
                    margin-bottom:8px;
                "
            >
                ANALYSIS SETTINGS
            </div>
            """
        )


        confidence = st.slider(
            "YOLO Confidence",
            min_value=0.10,
            max_value=0.90,
            value=0.25,
            step=0.05,
        )


        iou = st.slider(
            "YOLO IoU",
            min_value=0.10,
            max_value=0.90,
            value=0.45,
            step=0.05,
        )


        st.divider()


        st.html(
            """
            <div
                style="
                    color:#ff6b91;
                    font-size:11px;
                    letter-spacing:2px;
                    font-weight:700;
                    margin-bottom:12px;
                "
            >
                MODELS
            </div>
            """
        )


        st.write(
            "🔬 YOLOv11s"
        )

        st.caption(
            "yolo11s_bccd_best.pt"
        )


        st.write(
            "🧬 ConvNeXt-Tiny"
        )

        st.caption(
            f"Fold {classifier.SELECTED_FOLD}"
        )


        st.write(
            "🧠 CPU inference"
        )


        st.divider()


        st.caption(
            "Low-memory deployment configuration"
        )


    # ========================================================
    # UPLOAD AREA
    # ========================================================

    st.html(
        """
        <div
            style="
                margin-top:20px;
                margin-bottom:10px;
            "
        >

            <div
                style="
                    font-size:12px;
                    color:#ff6b91;
                    letter-spacing:2px;
                    font-weight:700;
                "
            >
                STEP 01
            </div>

            <div
                style="
                    font-size:28px;
                    font-weight:800;
                    margin-top:5px;
                "
            >
                Upload Blood Smear
            </div>

            <div
                style="
                    color:#858694;
                    margin-top:4px;
                "
            >
                Upload a microscopy image for
                automated blood-cell analysis.
            </div>

        </div>
        """
    )


    uploaded_file = st.file_uploader(
        "Choose microscopy image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "bmp",
            "tif",
            "tiff",
            "webp",
        ],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )


    if uploaded_file is None:

        st.html(
            """
            <div
                style="
                    text-align:center;
                    padding:55px 20px;
                    margin-top:10px;
                    border:1px dashed
                        rgba(255,255,255,0.15);
                    border-radius:22px;
                    background:
                        rgba(255,255,255,0.025);
                "
            >

                <div style="font-size:48px;">
                    🔬
                </div>

                <div
                    style="
                        font-size:20px;
                        font-weight:700;
                        margin-top:10px;
                    "
                >
                    Drop your blood smear image here
                </div>

                <div
                    style="
                        color:#777886;
                        margin-top:8px;
                    "
                >
                    JPG · PNG · BMP · TIFF · WEBP
                </div>

            </div>
            """
        )

        return


    # ========================================================
    # FILE SIZE
    # ========================================================

    file_size_mb = (
        uploaded_file.size
        / (1024 * 1024)
    )


    if file_size_mb > MAX_FILE_SIZE_MB:

        st.error(
            f"File is too large. Maximum "
            f"allowed size is "
            f"{MAX_FILE_SIZE_MB} MB."
        )

        return


    # ========================================================
    # READ IMAGE
    # ========================================================

    try:

        original_image = (
            Image.open(
                uploaded_file
            )
            .convert("RGB")
        )

    except Exception as exc:

        st.error(
            "Could not read the uploaded image."
        )

        st.code(
            str(exc)
        )

        return


    # ========================================================
    # PREVIEW
    # ========================================================

    st.html(
        """
        <div
            style="
                margin-top:30px;
                margin-bottom:12px;
            "
        >

            <div
                style="
                    font-size:12px;
                    color:#ff6b91;
                    letter-spacing:2px;
                    font-weight:700;
                "
            >
                STEP 02
            </div>

            <div
                style="
                    font-size:25px;
                    font-weight:800;
                    margin-top:5px;
                "
            >
                Image Preview
            </div>

        </div>
        """
    )


    st.image(
        original_image,
        caption=uploaded_file.name,
        use_container_width=True,
    )


    st.caption(
        f"{original_image.width} × "
        f"{original_image.height} pixels"
    )


    # ========================================================
    # ANALYZE BUTTON
    # ========================================================

    analyze = st.button(
        "🔬  Analyze Blood Smear",
        type="primary",
        use_container_width=True,
    )


    if not analyze:

        return


    temp_path = None


    try:

        # ====================================================
        # SAVE TEMP IMAGE
        # ====================================================

        suffix = Path(
            uploaded_file.name
        ).suffix


        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp_file:

            temp_file.write(
                uploaded_file.getbuffer()
            )

            temp_path = temp_file.name


        # ====================================================
        # YOLO DETECTION
        # ====================================================

        with st.spinner(
            "Detecting WBCs, RBCs and platelets..."
        ):

            detection_result = (
                detector.detect_image(
                    image_path=temp_path,
                    conf=confidence,
                    iou=iou,
                )
            )


        counts = detection_result.get(
            "counts",
            {},
        )


        detections = detection_result.get(
            "detections",
            [],
        )

        # ====================================================
        # WBC IMAGE VALIDATION
        # Reject unrelated images before subtype classification.
        # ====================================================
        valid_wbc_detections = [
            detection
            for detection in detections
            if (
                detection.get("class_name") == "WBC"
                and float(detection.get("confidence", 0.0))
                >= MIN_WBC_VALIDATION_CONFIDENCE
            )
        ]

        if not valid_wbc_detections:
            st.warning(
                "This image does not appear to contain a detectable white blood cell."
            )
            st.info(
                "Please upload a blood smear microscopy image containing white blood cells."
            )
            st.session_state.analysis_results = None
            return


        # ====================================================
        # CREATE WBC CROPS
        #
        # Current detector.py provides:
        # detect_image()
        # crop_wbc()
        #
        # So we create the crop list here.
        # ====================================================

        wbc_crops = []


        wbc_index = 1


        for detection in valid_wbc_detections:

            if (
                detection.get(
                    "class_name"
                )
                != "WBC"
            ):

                continue


            try:

                bbox = detection.get(
                    "bbox",
                    {},
                )

                # Do the crop here instead of calling detector.crop_wbc().
                # This keeps the Streamlit app compatible with detector.py
                # versions that use either image_path or image as input.
                crop = crop_wbc_from_image(
                    original_image,
                    bbox,
                    padding=0.15,
                )

                wbc_crops.append(
                    {
                        "wbc_index":
                            wbc_index,

                        "confidence":
                            detection.get(
                                "confidence",
                                0.0,
                            ),

                        "bbox":
                            bbox,

                        "crop":
                            crop,
                    }
                )

                wbc_index += 1

            except Exception as exc:

                print(
                    "[Streamlit] "
                    f"WBC crop failed: {exc}"
                )


        # ====================================================
        # SAVE RESULTS
        # ====================================================

        st.session_state.analysis_results = {
            "original_image":
                original_image,

            "counts":
                counts,

            "detections":
                detections,

            "wbc_crops":
                wbc_crops,

            "filename":
                uploaded_file.name,
        }


        # ====================================================
        # DETECTION SUMMARY
        # ====================================================

        st.html(
            """
            <div
                style="
                    margin-top:35px;
                    margin-bottom:15px;
                "
            >

                <div
                    style="
                        font-size:12px;
                        color:#ff6b91;
                        letter-spacing:2px;
                        font-weight:700;
                    "
                >
                    STEP 03
                </div>

                <div
                    style="
                        font-size:28px;
                        font-weight:800;
                        margin-top:5px;
                    "
                >
                    Detection Results
                </div>

            </div>
            """
        )


        col1, col2, col3, col4 = (
            st.columns(4)
        )


        with col1:

            st.metric(
                "White Blood Cells",
                counts.get(
                    "WBC",
                    0,
                ),
            )


        with col2:

            st.metric(
                "Red Blood Cells",
                counts.get(
                    "RBC",
                    0,
                ),
            )


        with col3:

            st.metric(
                "Platelets",
                counts.get(
                    "Platelets",
                    0,
                ),
            )


        with col4:

            st.metric(
                "Total Cells",
                len(detections),
            )


        # ====================================================
        # ANNOTATED IMAGE
        # ====================================================

        annotated_image = draw_detections(
            original_image,
            detections,
        )


        st.html(
            """
            <div
                style="
                    margin-top:30px;
                    margin-bottom:12px;
                "
            >

                <div
                    style="
                        font-size:12px;
                        color:#ff6b91;
                        letter-spacing:2px;
                        font-weight:700;
                    "
                >
                    YOLO DETECTION
                </div>

                <div
                    style="
                        font-size:24px;
                        font-weight:800;
                        margin-top:5px;
                    "
                >
                    Detected Blood Cells
                </div>

            </div>
            """
        )


        st.image(
            annotated_image,
            width=800,
        )


        # ====================================================
        # NO WBC
        # ====================================================

        if not wbc_crops:

            st.warning(
                "No WBCs were detected. "
                "WBC subtype classification "
                "was not performed."
            )

            return


        # ====================================================
        # CLASSIFICATION
        # ====================================================

        st.html(
            """
            <div
                style="
                    margin-top:40px;
                    margin-bottom:15px;
                "
            >

                <div
                    style="
                        font-size:12px;
                        color:#ff6b91;
                        letter-spacing:2px;
                        font-weight:700;
                    "
                >
                    STEP 04
                </div>

                <div
                    style="
                        font-size:28px;
                        font-weight:800;
                        margin-top:5px;
                    "
                >
                    WBC Subtype Classification
                </div>

                <div
                    style="
                        color:#858694;
                        margin-top:4px;
                    "
                >
                    ConvNeXt-Tiny · Fold
                    """
            + str(
                classifier.SELECTED_FOLD
            )
            + """
                </div>

            </div>
            """
        )


        with st.spinner(
            "Classifying detected WBCs..."
        ):

            classification_results = (
                classify_detected_wbcs(
                    wbc_crops
                )
            )


        # ====================================================
        # CLASSIFICATION CARDS
        # ====================================================

        for item in classification_results:

            index = item[
                "wbc_index"
            ]


            crop = item.get(
                "crop"
            )


            result = item.get(
                "classification",
                {},
            )


            st.html(
                "<div style='"
                "background:rgba(255,255,255,0.035);"
                "border:1px solid rgba(255,255,255,0.10);"
                "border-radius:22px;"
                "padding:22px;"
                "margin:15px 0;"
                "'>"
            )


            col_img, col_info = (
                st.columns(
                    [1, 2]
                )
            )


            with col_img:

                if crop is not None:

                    st.image(
                        crop,
                        caption=
                            f"WBC #{index}",
                        use_container_width=True,
                    )


            with col_info:

                if "error" in result:

                    st.error(
                        "Classification failed"
                    )

                    st.code(
                        result[
                            "error"
                        ]
                    )

                else:

                    class_name = (
                        result.get(
                            "class_name",
                            result.get(
                                "subtype",
                                "UNKNOWN",
                            ),
                        )
                    )


                    confidence_value = float(
                        result.get(
                            "confidence",
                            0.0,
                        )
                    )


                    st.html(
                        f"""
                        <div
                            style="
                                color:#888994;
                                font-size:11px;
                                letter-spacing:2px;
                                font-weight:700;
                            "
                        >
                            WBC #{index}
                        </div>

                        <div
                            style="
                                font-size:36px;
                                font-weight:800;
                                margin-top:5px;
                            "
                        >
                            {class_name}
                        </div>
                        """
                    )


                    st.write(
                        "Classification confidence"
                    )


                    st.progress(
                        min(
                            max(
                                confidence_value,
                                0.0,
                            ),
                            1.0,
                        )
                    )


                    st.html(
                        f"""
                        <div
                            style="
                                font-size:22px;
                                font-weight:700;
                                color:#ff789d;
                            "
                        >
                            {confidence_value * 100:.2f}%
                        </div>
                        """
                    )


                    detection_confidence = (
                        float(
                            item.get(
                                "detection_confidence",
                                0.0,
                            )
                        )
                    )


                    st.caption(
                        "YOLO detection confidence: "
                        f"{detection_confidence * 100:.2f}%"
                    )


                    reliability = result.get(
                        "reliability",
                        "unknown",
                    )


                    reliable = result.get(
                        "reliable",
                        False,
                    )


                    if reliable:

                        st.success(
                            "Classification reliability: "
                            f"{str(reliability).upper()}"
                        )

                    else:

                        st.warning(
                            "Classification reliability: "
                            f"{str(reliability).upper()}"
                        )


            st.html(
                "</div>",
            
            )


        # ====================================================
        # CLASSIFICATION SUMMARY
        # ====================================================

        st.html(
            """
            <div
                style="
                    margin-top:40px;
                    margin-bottom:15px;
                "
            >

                <div
                    style="
                        font-size:12px;
                        color:#ff6b91;
                        letter-spacing:2px;
                        font-weight:700;
                    "
                >
                    SUMMARY
                </div>

                <div
                    style="
                        font-size:28px;
                        font-weight:800;
                        margin-top:5px;
                    "
                >
                    Classification Summary
                </div>

            </div>
            """
        )


        subtype_counts: Dict[
            str,
            int,
        ] = {}


        for item in classification_results:

            result = item.get(
                "classification",
                {},
            )


            if "error" in result:

                continue


            subtype = result.get(
                "class_name",
                result.get(
                    "subtype",
                    "UNKNOWN",
                ),
            )


            subtype_counts[
                subtype
            ] = (
                subtype_counts.get(
                    subtype,
                    0,
                )
                + 1
            )


        if subtype_counts:

            summary_columns = st.columns(
                min(
                    len(subtype_counts),
                    4,
                )
            )


            for index, (
                subtype,
                count,
            ) in enumerate(
                sorted(
                    subtype_counts.items()
                )
            ):

                with summary_columns[
                    index
                    % len(summary_columns)
                ]:

                    st.metric(
                        subtype,
                        count,
                    )


        # ====================================================
        # SUCCESS
        # ====================================================

        st.success(
            "Analysis completed successfully."
        )


    except Exception as exc:

        st.error(
            "Analysis failed."
        )

        st.exception(
            exc
        )


    finally:

        # ----------------------------------------------------
        # TEMP FILE CLEANUP
        # ----------------------------------------------------

        if temp_path is not None:

            try:

                Path(
                    temp_path
                ).unlink(
                    missing_ok=True
                )

            except Exception:

                pass


        gc.collect()


# ============================================================
# ROUTER
# ============================================================

if st.session_state.page == "home":

    render_home()

else:

    render_analysis()


# ============================================================
# FOOTER
# ============================================================

st.html(
    """
    <div
        style="
            text-align:center;
            color:#555663;
            font-size:12px;
            padding:35px 0 10px 0;
        "
    >
        Blood Cell Intelligence
        · YOLOv11 Detection
        · ConvNeXt-Tiny Classification
    </div>
    """,
)