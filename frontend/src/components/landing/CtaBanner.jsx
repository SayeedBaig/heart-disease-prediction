/**
 * CtaBanner — final call-to-action banner with gradient background.
 * Preserves: /get-started link.
 */
import { ArrowRight } from "lucide-react";
import Section from "../ui/Section";
import Container from "../ui/Container";
import Button from "../ui/Button";

export default function CtaBanner() {
  return (
    <Section bg="primary">
      <Container>
        <div className="flex flex-col items-center justify-center rounded-3xl border border-[var(--accent-melanzane-border)] bg-gradient-to-r from-[var(--accent-melanzane-light)] via-[var(--card-bg)] to-[var(--accent-melanzane-light)] px-8 py-14 text-center shadow-xl md:px-16 md:py-20">
          <h2
            className="text-3xl font-extrabold tracking-tight text-[var(--text-primary)] md:text-4xl"
            style={{ fontFamily: "var(--font-display, 'Plus Jakarta Sans', sans-serif)" }}
          >
            Ready to evaluate cardiovascular health?
          </h2>
          <p className="mt-4 max-w-2xl text-base leading-relaxed text-[var(--text-secondary)]">
            Experience multimodal AI heart disease screening, explainable diagnostic reports, and interactive Digital Twin simulations.
          </p>
          <div className="mt-8 flex w-full justify-center">
            <Button to="/get-started" size="lg">
              Get Started Now
              <ArrowRight size={16} />
            </Button>
          </div>
        </div>
      </Container>
    </Section>
  );
}