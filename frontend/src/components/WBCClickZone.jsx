import { motion } from "framer-motion";

function WBCClickZone({ onClick }) {
  return (
    <motion.button
      className="wbc-click-zone"
      onClick={onClick}
      aria-label="Enter WBC analysis"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 1.5 }}
      whileHover={{
        scale: 1.04,
      }}
      whileTap={{
        scale: 0.97,
      }}
    >
      <span className="wbc-ring"></span>

      <span className="wbc-hover-text">
        ANALYZE
      </span>
    </motion.button>
  );
}

export default WBCClickZone;