/**
 * FaqSection — stateful accordion FAQ.
 * Preserves: openFaq state, toggle behavior, AnimatePresence.
 */
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronDown } from "lucide-react";
import Section from "../ui/Section";
import Container from "../ui/Container";
import SectionHeading from "../ui/SectionHeading";
import { fadeUp } from "../ui/animations";
import { faqs } from "./landingData";

export default function FaqSection() {
  const [openFaq, setOpenFaq] = useState(null);

  const toggleFaq = (index) => {
    setOpenFaq(openFaq === index ? null : index);
  };

  return (
    <Section id="faq" bg="card" bordered className="scroll-mt-20">
      <Container>
        <SectionHeading
          badge="FAQ"
          title="Frequently Asked Questions"
        />

        <div className="mt-14 flex flex-col gap-5">
          {faqs.map((item, index) => {
            const isOpen = openFaq === index;

            return (
              <motion.div
                key={item.question}
                {...fadeUp}
                className={`cursor-pointer rounded-2xl border bg-[var(--card-bg)] p-6 shadow-sm transition-all md:p-8 ${
                  isOpen
                    ? "border-[var(--accent-melanzane)] shadow-md"
                    : "border-[var(--border-color)] hover:border-[var(--accent-melanzane-border)] hover:shadow-md"
                }`}
                onClick={() => toggleFaq(index)}
              >
                <div className="flex select-none items-center justify-between gap-4">
                  <span className="text-base font-bold text-[var(--text-primary)] md:text-lg">
                    {item.question}
                  </span>
                  <div
                    className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-xl transition-all ${
                      isOpen
                        ? "bg-[var(--accent-melanzane)] text-white"
                        : "bg-[var(--accent-melanzane-light)] text-[var(--accent-melanzane)]"
                    }`}
                  >
                    <ChevronDown
                      size={18}
                      className={`transition-transform duration-300 ${
                        isOpen ? "rotate-180" : ""
                      }`}
                    />
                  </div>
                </div>

                <AnimatePresence>
                  {isOpen && (
                    <motion.p
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      exit={{ opacity: 0, height: 0 }}
                      transition={{ duration: 0.25 }}
                      className="mt-4 border-t border-[var(--border-subtle)] pt-4 text-sm leading-relaxed text-[var(--text-secondary)] md:text-base"
                    >
                      {item.answer}
                    </motion.p>
                  )}
                </AnimatePresence>
              </motion.div>
            );
          })}
        </div>
      </Container>
    </Section>
  );
}