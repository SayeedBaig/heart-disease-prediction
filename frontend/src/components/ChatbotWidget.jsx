import { useState, useEffect, useRef } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Bot, MessageCircle, Send, X } from "lucide-react";

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

// Read the last prediction result from localStorage so the chatbot can
// include risk context in every request (avoids asking the user to repeat it).
function getPredictionContext() {
  try {
    const result = JSON.parse(localStorage.getItem("prediction_result") || "{}");
    return {
      risk_level: result?.fusion?.final_level || "",
      risk_percentage: result?.fusion?.risk_percentage || 0,
      ecg_class: result?.ecg?.class || "",
      ef_value: result?.echo?.ef_value || 0,
    };
  } catch {
    return { risk_level: "", risk_percentage: 0, ecg_class: "", ef_value: 0 };
  }
}

// Akash — patient conversation memory needs a stable session_id across
// messages (and ideally across a browser session). Generate once, reuse.
function getPatientSessionId() {
  let sessionId = localStorage.getItem("cardio-chat-session");
  if (!sessionId) {
    sessionId = `patient-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
    localStorage.setItem("cardio-chat-session", sessionId);
  }
  return sessionId;
}

// Akash — mirrors doctorService.js's doctorFetch token resolution exactly,
// so doctor-mode auth behaves identically to the rest of the doctor portal.
function getAuthToken(mode) {
  if (mode === "doctor") {
    return (
      localStorage.getItem("doctor_token") ||
      localStorage.getItem("doctor_access_token") ||
      localStorage.getItem("access_token")
    );
  }
  return localStorage.getItem("access_token");
}

export default function ChatbotWidget({ mode = "patient" }) {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const scrollRef = useRef(null);

  const endpoint = "/rag/ask"; // single endpoint handles both modes
  const headerTitle = mode === "doctor" ? "Clinical AI Assistant" : "AI Health Assistant";

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isOpen]);

  const handleSend = async () => {
    const question = input.trim();
    if (!question || isLoading) return;

    const userMessage = { id: Date.now(), sender: "user", text: question };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    // Akash — build the request body per mode. Patient mode adds session_id
    // for conversation memory; doctor mode sends the case context fields
    // the backend's history_context_builder.py expects.
    const context = getPredictionContext();
    // /rag/ask accepts: { question, context?: { risk_level, risk_percentage, ecg_class, ef_value } }
    const body = {
      question,
      context: {
        risk_level: context.risk_level || null,
        risk_percentage: context.risk_percentage || null,
        ecg_class: context.ecg_class || null,
        ef_value: context.ef_value || null,
      },
    };

    try {
      const token = getAuthToken(mode);
      const headers = { "Content-Type": "application/json" };
      if (token) headers.Authorization = `Bearer ${token}`;

      const res = await fetch(`${BASE_URL}${endpoint}`, {
        method: "POST",
        headers,
        body: JSON.stringify(body),
      });

      const data = await res.json().catch(() => ({}));

      if (!res.ok) {
        throw new Error(data?.detail || "Request failed");
      }

      const aiMessage = {
        id: Date.now() + 1,
        sender: "ai",
        text: data.answer || "Sorry, I didn't get a response for that.",
      };
      setMessages((prev) => [...prev, aiMessage]);
    } catch (err) {
      console.error("ChatbotWidget error:", err);
      const errorMessage = {
        id: Date.now() + 1,
        sender: "ai",
        text: "Sorry, I couldn't reach the AI Assistant right now. Please try again.",
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="fixed bottom-6 left-6 z-50 flex flex-col items-start">
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: 16, scale: 0.96 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 16, scale: 0.96 }}
            transition={{ duration: 0.18 }}
            className="cardio-card mb-3 w-[340px] h-[440px] flex flex-col overflow-hidden shadow-2xl border border-[var(--border-color)] rounded-2xl"
          >
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-[var(--border-color)] bg-[var(--card-bg)]">
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-xl bg-[var(--accent-melanzane-light)] text-[var(--accent-melanzane)] flex items-center justify-center">
                  <Bot size={18} />
                </div>
                <span className="text-sm font-bold text-[var(--text-primary)]">
                  {headerTitle}
                </span>
              </div>
              <button
                onClick={() => setIsOpen(false)}
                className="text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
                aria-label="Close chat"
              >
                <X size={18} />
              </button>
            </div>

            {/* Messages */}
            <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-3">
              {messages.length === 0 ? (
                <div className="h-full flex flex-col items-center justify-center text-center px-2">
                  <Bot size={28} className="text-[var(--accent-melanzane)] mb-2" />
                  <p className="text-xs text-[var(--text-secondary)]">
                    Ask about {mode === "doctor" ? "this case, guidelines, or visit history" : "your results, food, or heart health"}.
                  </p>
                </div>
              ) : (
                messages.map((msg) => (
                  <div
                    key={msg.id}
                    className={`flex ${msg.sender === "user" ? "justify-end" : "justify-start"}`}
                  >
                    <div
                      className={`max-w-[80%] rounded-2xl px-3 py-2 text-xs leading-relaxed ${
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
              {isLoading && (
                <div className="flex justify-start">
                  <div className="rounded-2xl px-3 py-2 text-xs bg-[var(--bg-secondary)] border border-[var(--border-color)] text-[var(--text-secondary)]">
                    Thinking…
                  </div>
                </div>
              )}
            </div>

            {/* Input */}
            <div className="p-3 border-t border-[var(--border-color)] bg-[var(--card-bg)]">
              <div className="flex items-center gap-2">
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") handleSend();
                  }}
                  placeholder="Type a message..."
                  className="cardio-input text-xs flex-1"
                  disabled={isLoading}
                />
                <button
                  onClick={handleSend}
                  disabled={!input.trim() || isLoading}
                  className="btn-primary text-xs p-2.5 rounded-xl disabled:opacity-50"
                  aria-label="Send"
                >
                  <Send size={14} />
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Launcher bubble */}
      <button
        onClick={() => setIsOpen((prev) => !prev)}
        className="w-14 h-14 rounded-full bg-[var(--accent-melanzane)] text-white flex items-center justify-center shadow-lg hover:scale-105 transition-transform"
        aria-label="Open AI Assistant"
      >
        {isOpen ? <X size={22} /> : <MessageCircle size={22} />}
      </button>
    </div>
  );
}
