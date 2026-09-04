import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";

function AnalysisPage() {
  const navigate = useNavigate();

  return (
    <motion.main
      className="analysis-page"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.8 }}
    >

      <header className="analysis-header">

        <div>
          <p className="analysis-eyebrow">
            AI HEMATOLOGY PLATFORM
          </p>

          <h1>
            Blood Cell Analysis
          </h1>
        </div>

        <button
          className="back-button"
          onClick={() => navigate("/")}
        >
          ← BACK
        </button>

      </header>


      <section className="analysis-container">

        <div className="analysis-intro">

          <p className="section-label">
            ANALYSIS WORKSPACE
          </p>

          <h2>
            Upload a blood image
            <br />
            for AI-assisted analysis
          </h2>

          <p>
            The system will analyze detected blood cells,
            WBC morphology and image-derived AML evidence.
          </p>

        </div>


        <div className="upload-card">

          <div className="upload-icon">
            +
          </div>

          <h3>
            Upload Blood Smear Image
          </h3>

          <p>
            JPG, JPEG or PNG
          </p>

          <button className="upload-button">
            SELECT IMAGE
          </button>

        </div>


        <div className="feature-grid">

          <div className="feature-card">
            <span>01</span>
            <h3>Cell Detection</h3>
            <p>
              Detection and counting of RBCs,
              WBCs and platelets.
            </p>
          </div>

          <div className="feature-card">
            <span>02</span>
            <h3>WBC Analysis</h3>
            <p>
              AI-based WBC subtype assessment
              using the trained ensemble.
            </p>
          </div>

          <div className="feature-card">
            <span>03</span>
            <h3>AML Evidence</h3>
            <p>
              Image-model assessment for
              AML-related cytomorphology.
            </p>
          </div>

          <div className="feature-card">
            <span>04</span>
            <h3>Interpretability</h3>
            <p>
              Visual model attention through
              Grad-CAM analysis.
            </p>
          </div>

        </div>

      </section>

      <footer className="clinical-note">

        <strong>RESEARCH PROTOTYPE</strong>

        <span>
          This system provides image-derived AI evidence
          and does not constitute a medical diagnosis.
        </span>

      </footer>

    </motion.main>
  );
}

export default AnalysisPage;