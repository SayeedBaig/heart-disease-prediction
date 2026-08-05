/**
 * StatsBar — four-up metric grid below the hero.
 * Each card uses the shared Card component for equal height.
 */
import { motion } from "framer-motion";
import Section from "../ui/Section";
import Container from "../ui/Container";
import Grid from "../ui/Grid";
import Card from "../ui/Card";
import { fadeUp } from "../ui/animations";
import { stats } from "./landingData";

export default function StatsBar() {
  return (
    <Section bg="card" bordered>
      <Container>
        <Grid cols={4} gap="gap-6">
          {stats.map((item) => (
            <motion.div key={item.label} {...fadeUp} className="h-full">
              <Card className="text-center">
                <div>
                  <p className="text-3xl font-extrabold text-[var(--accent-melanzane)] md:text-4xl">
                    {item.value}
                  </p>
                  <p className="mt-2 text-sm font-bold text-[var(--text-primary)]">
                    {item.label}
                  </p>
                  <p className="mt-1 text-xs text-[var(--text-secondary)]">
                    {item.desc}
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
