/**
 * CapabilitiesSection — six-card grid of platform features.
 * Uses Section, SectionHeading, Grid, and Card for consistent layout.
 */
import { motion } from "framer-motion";
import Section from "../ui/Section";
import Container from "../ui/Container";
import SectionHeading from "../ui/SectionHeading";
import Grid from "../ui/Grid";
import Card from "../ui/Card";
import { fadeUp } from "../ui/animations";
import { capabilities } from "./landingData";

export default function CapabilitiesSection() {
  return (
    <Section id="capabilities" bg="primary" bordered className="scroll-mt-20">
      <Container>
        <SectionHeading
          badge="Platform Capabilities"
          title="Comprehensive Cardiac Intelligence Suite"
          subtitle="Designed for clinical accuracy, seamless data integration, and intuitive diagnostic visualization across every step of cardiovascular care."
        />

        <Grid cols={3} gap="gap-8" className="mt-16">
          {capabilities.map(({ icon: Icon, title, description }, index) => (
            <motion.div
              key={title}
              {...fadeUp}
              transition={{ duration: 0.4, delay: index * 0.05 }}
              className="h-full"
            >
              <Card interactive className="group">
                <div>
                  <div className="mb-6 flex h-14 w-14 items-center justify-center rounded-2xl border border-[var(--accent-melanzane-border)] bg-[var(--accent-melanzane-light)] text-[var(--accent-melanzane)] transition-transform group-hover:scale-110 group-hover:bg-[var(--accent-melanzane)] group-hover:text-white">
                    <Icon size={26} />
                  </div>
                  <h3 className="text-xl font-extrabold leading-snug text-[var(--text-primary)] transition-colors group-hover:text-[var(--accent-melanzane)]">
                    {title}
                  </h3>
                  <p className="mt-3 text-sm leading-relaxed text-[var(--text-secondary)]">
                    {description}
                  </p>
                </div>
              </Card>
            </motion.div>
          ))}
        </Grid>
      </Container>
    </Section>
  );
}