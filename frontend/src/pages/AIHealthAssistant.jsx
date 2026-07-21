import api from "../services/api";

import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowLeft, Bot, Send } from "lucide-react";

function AIHealthAssistant() {
  const navigate = useNavigate();

  // Input field
  const [message, setMessage] = useState("");

  // Chat messages
  const [messages, setMessages] = useState([]);

  const result = JSON.parse(
  localStorage.getItem("prediction_result") || "{}"
);

  // Suggested prompts
  const suggestions = [
    "Explain ECG Findings",
    "Summarize Patient Risk",
    "Lifestyle Recommendations",
    "ACC/AHA Guidelines",
  ];

  const handleSend = async () => {
  if (!message.trim()) return;

  const userMessage = {
  id: Date.now(),
  sender: "user",
  text: message,
};

  setMessages((prev) => [...prev, userMessage]);

  const currentQuestion = message;
  setMessage("");

  try {
    const response = await api.post("/rag/ask", {
  question: currentQuestion,
  context: {
    risk_level: result?.fusion?.final_level || "",
    risk_percentage: result?.fusion?.risk_percentage || 0,
    ecg_class: result?.ecg?.level || "",
    ef_value: 0,
  },
});

    const aiMessage = {
  id: Date.now() + 1,
  sender: "ai",
  text: response.data.answer,
};

    setMessages((prev) => [...prev, aiMessage]);
  } catch (error) {
    const errorMessage = {
  id: Date.now() + 1,
  sender: "ai",
  text: "Sorry, I couldn't connect to the AI Assistant. Please try again.",
};

    setMessages((prev) => [...prev, errorMessage]);

    console.error(error);
  }
};
    /*
      TODO (Backend Integration)

      const response = await api.post("/assistant", {
          question: message,
          prediction_id: ...,
      });

      setMessages(prev => [
          ...prev,
          {
              id: Date.now()+1,
              sender:"ai",
              text: response.data.answer
          }
      ]);
    */

  // Click on suggested question
  const handleSuggestionClick = (question) => {
    setMessage(question);
  };

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <div className="max-w-6xl mx-auto">

        {/* ================= HEADER ================= */}

        <div className="mb-6">

          <button
            onClick={() => navigate("/doctor/dashboard")}
            className="flex items-center gap-2 text-blue-600 hover:text-blue-800 font-medium mb-4"
          >
            <ArrowLeft size={18} />
            Back to Dashboard
          </button>

          <h1 className="text-4xl font-bold text-gray-800">
  AI Health Assistant
</h1>

<p className="text-gray-600 mt-2 text-lg">
  Ask AI to explain heart disease predictions,
  ECG findings, clinical risk,
  and medical guidelines.
</p>

<div className="mt-4">
  <button
    onClick={() => setMessages([])}
    className="bg-red-600 text-white px-4 py-2 rounded-lg hover:bg-red-700"
  >
    🗑 Clear Chat
  </button>
</div>

        </div>

        {/* Suggested Questions */}

        <div className="bg-white rounded-xl shadow-md p-5 mb-6">

          <h2 className="text-lg font-semibold mb-4">
            Suggested Questions
          </h2>

          <div className="flex flex-wrap gap-3">

            {suggestions.map((item, index) => (
              <button
                key={index}
                onClick={() => handleSuggestionClick(item)}
                className="px-4 py-2 rounded-full border border-blue-300 bg-blue-50 text-blue-700 hover:bg-blue-100 transition"
              >
                {item}
              </button>
            ))}

          </div>

        </div>
                {/* ================= CHAT AREA ================= */}

        <div className="bg-white rounded-xl shadow-md h-[430px] flex flex-col mb-6">

          <div className="flex-1 overflow-y-auto p-6">

            {messages.length === 0 ? (

              <div className="h-full flex flex-col items-center justify-center text-center">

                <div className="bg-blue-100 p-5 rounded-full mb-5">
                  <Bot className="text-blue-600" size={42} />
                </div>

                <h2 className="text-2xl font-bold text-gray-800 mb-2">
                  CardioAI Health Assistant
                </h2>

                <p className="text-gray-500 max-w-lg mb-8">
                  Ask questions about prediction results, ECG findings,
                  cardiovascular risk, lifestyle recommendations,
                  or clinical guidelines.
                </p>

                <div className="grid grid-cols-2 gap-3 text-left">

                  <div className="bg-gray-50 rounded-lg px-4 py-3 border">
                    ❤️ Explain Prediction
                  </div>

                  <div className="bg-gray-50 rounded-lg px-4 py-3 border">
                    📈 Explain ECG Findings
                  </div>

                  <div className="bg-gray-50 rounded-lg px-4 py-3 border">
                    🥗 Lifestyle Advice
                  </div>

                  <div className="bg-gray-50 rounded-lg px-4 py-3 border">
                    📚 Medical Guidelines
                  </div>

                </div>

              </div>

            ) : (

              <div className="space-y-5">

                {messages.map((msg) => (

                  <div
                    key={msg.id}
                    className={`flex ${
                      msg.sender === "user"
                        ? "justify-end"
                        : "justify-start"
                    }`}
                  >

                    <div
                      className={`max-w-[75%] rounded-2xl px-5 py-3 shadow-sm ${
                        msg.sender === "user"
                          ? "bg-blue-600 text-white rounded-br-md"
                          : "bg-gray-100 text-gray-800 rounded-bl-md"
                      }`}
                    >

                      <p className="text-sm leading-7">
                        {msg.text}
                      </p>

                    </div>

                  </div>

                ))}

              </div>

            )}

          </div>

        </div>
                {/* ================= INPUT AREA ================= */}

        <div className="bg-white rounded-xl shadow-md p-4">
          <div className="flex gap-3">

            <input
              type="text"
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  handleSend();
                }
              }}
              placeholder="Ask about prediction results, ECG findings, or medical guidelines..."
              className="flex-1 border border-gray-300 rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />

            <button
  onClick={handleSend}
  disabled={!message.trim()}
  className={`px-6 rounded-xl flex items-center gap-2 transition text-white ${
    message.trim()
      ? "bg-blue-600 hover:bg-blue-700"
      : "bg-gray-400 cursor-not-allowed"
  }`}
>
              <Send size={18} />
              Send
            </button>

          </div>
        </div>

      </div>
    </div>
  );
}

export default AIHealthAssistant;