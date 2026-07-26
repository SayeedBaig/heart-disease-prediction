import { ShieldCheck, AlertTriangle, ShieldAlert } from "lucide-react";

function RiskBadge({ level }) {
  let styles = {
    bg: "bg-gray-100",
    text: "text-gray-700",
    border: "border-gray-200",
    icon: <ShieldCheck size={18} />,
  };

  switch (level) {
    case "Low":
      styles = {
        bg: "bg-green-50",
        text: "text-green-700",
        border: "border-green-200",
        icon: <ShieldCheck size={18} />,
      };
      break;

    case "Medium":
      styles = {
        bg: "bg-yellow-50",
        text: "text-yellow-700",
        border: "border-yellow-200",
        icon: <AlertTriangle size={18} />,
      };
      break;

    case "High":
      styles = {
        bg: "bg-red-50",
        text: "text-red-700",
        border: "border-red-200",
        icon: <ShieldAlert size={18} />,
      };
      break;

    default:
      break;
  }

  return (
    <span
      className={`
        inline-flex
        items-center
        gap-2
        ${styles.bg}
        ${styles.text}
        border
        ${styles.border}
        px-4
        py-2
        rounded-full
        font-semibold
        text-sm
        shadow-sm
        transition-all
        duration-300
        cursor-default
      `}
    >
      {styles.icon}
      {level} Risk
    </span>
  );
}

export default RiskBadge;