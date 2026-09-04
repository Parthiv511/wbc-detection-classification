import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";

import WBCClickZone from "../components/WBCClickZone";
import bloodBackground from "../assets/blood-background.mp4";

function LandingPage() {
  const navigate = useNavigate();

  const handleWBCClick = () => {
    navigate("/analysis");
  };

  return (
    <motion.main
      className="landing-page"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 1 }}
    >
      {/* Blood cell animation */}
      <video
        className="background-video"
        autoPlay
        muted
        loop
        playsInline
        preload="auto"
      >
        <source
          src={bloodBackground}
          type="video/mp4"
        />
      </video>

      {/* Dark cinematic layer */}
      <div className="background-overlay" />

      {/* Main content */}
      <section className="landing-content">

        {/* Brand */}
        <motion.div
          className="brand-block"
          initial={{
            opacity: 0,
            y: 30,
          }}
          animate={{
            opacity: 1,
            y: 0,
          }}
          transition={{
            duration: 1.2,
            delay: 0.4,
          }}
        >
          <p className="eyebrow">
            AI-POWERED HEMATOLOGY
          </p>

          <h1>
            BloodCell
            <span>Intelligence</span>
          </h1>

          <p className="landing-description">
            AI-assisted blood cell analysis and
            leukemia-related image assessment.
          </p>
        </motion.div>

        {/* WBC interactive area */}
        <WBCClickZone
          onClick={handleWBCClick}
        />

        {/* Instruction */}
        <motion.p
          className="interaction-hint"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{
            duration: 1,
            delay: 2,
          }}
        >
          Explore the WBC
        </motion.p>

      </section>

      {/* Footer */}
      <motion.div
        className="landing-footer"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{
          duration: 1,
          delay: 2,
        }}
      >
        <span>AI-ASSISTED ANALYSIS</span>
        <span>•</span>
        <span>IMAGE-BASED EVIDENCE</span>
        <span>•</span>
        <span>RESEARCH PROTOTYPE</span>
      </motion.div>
    </motion.main>
  );
}

export default LandingPage;