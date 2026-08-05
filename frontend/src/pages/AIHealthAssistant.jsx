import api from "../services/api";
import { useState } from "react";
import { Bot, Send, Trash2, Sparkles } from "lucide-react";
import Navbar from "../components/Navbar";

function AIHealthAssistant() {
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState([]);

  const result = JSON.parse(
    localStorage.getItem("prediction_result") || "{}"
  );

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
      console.error(error);
      const errorMessage = {
        id: Date.now() + 1,
        sender: "ai",
        text: "Sorry, I couldn't connect to the AI Assistant. Please check backend connection.",
      };

      setMessages((prev) => [...prev, errorMessage]);
    }
  };

  const handleSuggestionClick = (question) => {
    setMessage(question);
  };

  return (
    <div className="cardio-shell">
      <Navbar breadcrumb="AI Health Assistant" />

      <main className="cardio-container flex-1">
        {/* Header */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between pb-6 mb-6 border-b border-[var(--border-color)] gap-4">
          <div>
            <div className="flex items-center gap-2">
              <Sparkles size={16} className="text-[var(--accent-melanzane)]" />
              <span className="caption-small text-[var(--accent-melanzane)] uppercase font-bold tracking-wider">
                Clinical Intelligence RAG Assistant
              </span>
            </div>
            <h1 className="h2-semibold text-[var(--text-primary)] mt-1">
              AI Health Assistant
            </h1>
            <p className="body-regular text-xs mt-1">
              Ask questions about prediction results, ECG waves, clinical risk, and ACC/AHA guidelines.
            </p>
          </div>

          <button
            onClick={() => setMessages([])}
            className="btn-secondary text-xs py-2 px-3 rounded-xl flex items-center gap-1.5 hover:text-red-500 hover:border-red-200 transition-all shrink-0"
          >
            <Trash2 size={14} />
            <span>Clear Chat</span>
          </button>
        </div>

        {/* Suggested Questions */}
        <div className="cardio-card p-5 mb-6">
          <h2 className="caption-small font-bold text-[var(--text-primary)] uppercase tracking-wider mb-3">
            Suggested Prompts
          </h2>
          <div className="flex flex-wrap gap-2.5">
            {suggestions.map((item, index) => (
              <button
                key={index}
                onClick={() => handleSuggestionClick(item)}
                className="py-1.5 px-3.5 rounded-xl border border-[var(--accent-melanzane-border)] bg-[var(--accent-melanzane-light)] text-[var(--accent-melanzane)] text-xs font-semibold hover:bg-[var(--accent-melanzane-border)] transition-all"
              >
                {item}
              </button>
            ))}
          </div>
        </div>

        {/* Chat Container */}
        <div className="cardio-card p-0 h-[450px] flex flex-col mb-6 overflow-hidden">
          <div className="flex-1 overflow-y-auto p-6 space-y-4">
            {messages.length === 0 ? (
              <div className="h-full flex flex-col items-center justify-center text-center p-4">
                <div className="w-16 h-16 rounded-2xl bg-[var(--accent-melanzane-light)] text-[var(--accent-melanzane)] flex items-center justify-center mb-4">
                  <Bot size={36} />
                </div>
                <h2 className="text-lg font-bold text-[var(--text-primary)] mb-1">
                  CardioAI Multimodal Intelligence Assistant
                </h2>
                <p className="caption-small max-w-md mb-6">
                  Query clinical parameters, ECG classifications, echocardiography EF metrics, or ACC/AHA guidelines.
                </p>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-left max-w-md w-full">
                  <div className="cardio-card p-3 text-xs font-semibold text-[var(--text-primary)] flex items-center gap-2">
                    <span>❤️</span> Explain Risk Assessment
                  </div>
                  <div className="cardio-card p-3 text-xs font-semibold text-[var(--text-primary)] flex items-center gap-2">
                    <span>📈</span> Explain ECG Waveform
                  </div>
                  <div className="cardio-card p-3 text-xs font-semibold text-[var(--text-primary)] flex items-center gap-2">
                    <span>🥗</span> Lifestyle Interventions
                  </div>
                  <div className="cardio-card p-3 text-xs font-semibold text-[var(--text-primary)] flex items-center gap-2">
                    <span>📚</span> ACC/AHA Guidelines
                  </div>
                </div>
              </div>
            ) : (
              messages.map((msg) => (
                <div
                  key={msg.id}
                  className={`flex ${
                    msg.sender === "user" ? "justify-end" : "justify-start"
                  }`}
                >
                  <div
                    className={`max-w-[80%] rounded-2xl px-4 py-3 text-xs leading-relaxed ${
                      msg.sender === "user"
                        ? "bg-[var(--accent-melanzane)] text-white rounded-br-none"
                        : "bg-[var(--bg-secondary)] text-[var(--text-primary)] border border-[var(--border-color)] rounded-bl-none"
                    }`}
                  >
                    {msg.text}
                  </div>
                </div>
              ))
            )}
          </div>

          {/* Input Bar */}
          <div className="p-4 border-t border-[var(--border-color)] bg-[var(--card-bg)]">
            <div className="flex items-center gap-3">
              <input
                type="text"
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleSend();
                }}
                placeholder="Ask about prediction results, ECG findings, or guidelines..."
                className="cardio-input text-xs flex-1"
              />
              <button
                onClick={handleSend}
                disabled={!message.trim()}
                className="btn-primary text-xs py-2.5 px-5 rounded-xl disabled:opacity-50 flex items-center gap-2"
              >
                <Send size={15} />
                <span>Send</span>
              </button>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default AIHealthAssistant;