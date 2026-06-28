function RiskBadge({ level }) {
  let color = "bg-gray-200 text-gray-800";

  if (level === "Low") {
    color = "bg-green-100 text-green-700";
  } else if (level === "Medium") {
    color = "bg-yellow-100 text-yellow-700";
  } else if (level === "High") {
    color = "bg-red-100 text-red-700";
  }

  return (
    <span
      className={`${color} px-3 py-1 rounded-full text-sm font-semibold`}
    >
      {level}
    </span>
  );
}

export default RiskBadge;