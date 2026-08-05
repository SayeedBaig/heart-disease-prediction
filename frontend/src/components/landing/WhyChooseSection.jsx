/**
 * WhyChooseSection — three-card grid of differentiators.
 * Uses Section, SectionHeading, Grid, and Card for consistent layout.
 */
import { motion } from "framer-motion";
import Section from "../ui/Section";
import Container from "../ui/Container";
import SectionHeading from "../ui/SectionHeading";
import Grid from "../ui/Grid";
import Card from "../ui/Card";
import { fadeUp } from "../ui/animations";
import { reasons } from "./landingData";

export default function WhyChooseSection() {
  return (
    <Section bg="primary">
      <Container>
        <SectionHeading
          badge="Why Choose CardioAI"
          title="Next-generation cardiovascular intelligence"
          subtitle="Bridging diagnostic medical data with modern artificial intelligence to deliver unprecedented clarity for healthcare providers and patients."
        />

        <Grid cols={3} gap="gap-8" className="mt-14">
          {reasons.map(({ icon: Icon, title, description }, index) => (
            <motion.div
              key={title}
              {...fadeUp}
              transition={{ duration: 0.4, delay: index * 0.06 }}
              className="h-full"
            >
              <Card interactive>
                <div>
                  <div className="mb-6 flex h-14 w-14 items-center justify-center rounded-2xl bg-[var(--accent-melanzane-light)] text-[var(--accent-melanzane)]">
                    <Icon size={26} />
                  </div>
                  <h3 className="text-lg font-bold text-[var(--text-primary)]">{title}</h3>
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