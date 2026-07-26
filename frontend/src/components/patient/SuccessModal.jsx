import { CheckCircle2 } from "lucide-react";

function SuccessModal({
  open,
  patientId,
  patientName,
  onContinue,
}) {
  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm">

      <div className="w-full max-w-md rounded-3xl bg-white p-8 shadow-2xl">

        <div className="flex justify-center">

          <div className="flex h-20 w-20 items-center justify-center rounded-full bg-green-100">

            <CheckCircle2
              className="text-green-600"
              size={50}
            />

          </div>

        </div>

        <h2 className="mt-6 text-center text-3xl font-bold text-slate-800">
          Patient Registered
        </h2>

        <p className="mt-3 text-center text-slate-500">
          The patient has been registered successfully.
        </p>

        <div className="mt-8 rounded-2xl bg-slate-50 p-5">

          <div className="mb-4">

            <p className="text-sm text-slate-500">
              Patient ID
            </p>

            <h3 className="text-xl font-bold text-blue-600">
              {patientId}
            </h3>

          </div>

          <div>

            <p className="text-sm text-slate-500">
              Patient Name
            </p>

            <h3 className="text-lg font-semibold">
              {patientName}
            </h3>

          </div>

        </div>

        <button
          onClick={onContinue}
          className="
            mt-8
            w-full
            rounded-xl
            bg-blue-600
            py-3
            text-lg
            font-semibold
            text-white
            transition
            hover:bg-blue-700
          "
        >
          Proceed to Diagnosis →
        </button>

      </div>

    </div>
  );
}

export default SuccessModal;