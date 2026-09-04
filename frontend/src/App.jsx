import { useNavigate } from "react-router-dom";
import "./index.css";

import bloodVideo from "./assets/blood-background.mp4";

function App() {
  const navigate = useNavigate();

  const handleWBCClick = () => {
    navigate("/analysis");
  };

  return (
    <main className="landing-page">

      {/* =====================================================
          BACKGROUND BLOOD-FLOW VIDEO
          ===================================================== */}
      <video
        className="hero-background"
        autoPlay
        muted
        loop
        playsInline
        preload="auto"
        aria-hidden="true"
      >
        <source src={bloodVideo} type="video/mp4" />
      </video>


      {/* =====================================================
          CINEMATIC OVERLAY
          ===================================================== */}
      <div className="background-overlay" />


      {/* =====================================================
          MAIN LANDING CONTENT
          ===================================================== */}
      <div className="landing-content">

        {/* ===================================================
            BRAND
            =================================================== */}
        <section className="brand-block">

          <div className="eyebrow">
            AI-POWERED HEMATOLOGY
          </div>

          <h1>
            <strong>BloodCell</strong>
            <span>Intelligence</span>
          </h1>

          <p className="landing-description">
            AI-assisted blood cell analysis and leukemia-related image
            assessment.
          </p>

        </section>


        {/* ===================================================
            WBC INTERACTIVE REGION

            ONLY THIS CIRCLE IS CLICKABLE.

            Clicking anywhere outside this circle does NOT
            navigate to the analysis page.
            =================================================== */}
        <button
          type="button"
          className="wbc-click-zone"
          onClick={handleWBCClick}
          aria-label="Analyze WBC"
        >

          {/* Outer analysis circle */}
          <div className="wbc-ring" />

          {/* Center text */}
          <div className="wbc-hover-text">
            ANALYZE WBC
          </div>

        </button>


        {/* ===================================================
            EXPLORE INDICATOR
            =================================================== */}
        <div className="interaction-hint">
          EXPLORE THE WBC
        </div>


        {/* ===================================================
            FOOTER
            =================================================== */}
        <div className="landing-footer">

          <span>AI-ASSISTED ANALYSIS</span>

          <span className="footer-dot">•</span>

          <span>IMAGE-BASED EVIDENCE</span>

          <span className="footer-dot">•</span>

          <span>RESEARCH PROTOTYPE</span>

        </div>

      </div>

    </main>
  );
}

export default App;