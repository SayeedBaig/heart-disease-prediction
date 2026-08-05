import { Link, useNavigate } from "react-router-dom";
import {
  Stethoscope,
  Activity,
  HeartPulse,
  Brain,
  ShieldCheck,
  FileText,
  UserCheck,
  Calendar,
  Layers,
  ArrowRight,
  ArrowLeft,
} from "lucide-react";

import Navbar from "../components/Navbar";
import Workflow from "../components/Workflow";
import { AppLayout } from "../components/ui/AppLayout";
import { Container } from "../components/ui/Container";
import { Section } from "../components/ui/Section";
import { Card, CardContent } from "../components/ui/Card";
import { Button } from "../components/ui/Button";

export default function ExploreCapabilities() {
  const navigate = useNavigate();

  return (
    <AppLayout>
      <Navbar onBack={() => navigate("/")} backLabel="Home" breadcrumb="Platform Capabilities" />

      <main className="flex-1 overflow-x-hidden">
        <Section className="py-12 md:py-16">
          <Container>
            {/* Back Button Banner */}
            <div className="mb-8 flex items-center justify-between border-b border-border pb-4">
              <Button
                variant="outline"
                size="sm"
                onClick={() => navigate("/")}
                className="gap-2"
              >
                <ArrowLeft size={16} />
                <span>Back to Home</span>
              </Button>

              <span className="text-xs text-primary font-bold uppercase tracking-wider">
                CardioAI Platform Capabilities
              </span>
            </div>

            {/* Hero Title for Capabilities */}
            <div className="text-center max-w-3xl mx-auto mb-16">
              <span className="text-xs uppercase tracking-widest text-primary font-bold">
                Platform Capabilities & Architecture
              </span>
              <h1 className="text-3xl sm:text-4xl md:text-5xl font-extrabold tracking-tight text-foreground mt-4">
                Comprehensive Cardiac Intelligence Suite
              </h1>
              <p className="mt-6 text-sm md:text-base text-muted-foreground leading-relaxed max-w-2xl mx-auto">
                Explore our multimodal neural architecture, clinical workflow, and interactive Digital Twin simulation engine.
              </p>
            </div>

            {/* Clean Statistics Row */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 w-full max-w-4xl mx-auto mb-20">
              <Card className="text-center p-8 border-border">
                <h3 className="text-4xl md:text-5xl font-extrabold text-primary mb-2">96%</h3>
                <p className="text-sm font-medium text-muted-foreground">Prediction Confidence</p>
              </Card>

              <Card className="text-center p-8 border-border">
                <h3 className="text-4xl md:text-5xl font-extrabold text-primary mb-2">3-in-1</h3>
                <p className="text-sm font-medium text-muted-foreground">Clinical • ECG • Echo Fusion</p>
              </Card>

              <Card className="text-center p-8 border-border">
                <h3 className="text-4xl md:text-5xl font-extrabold text-primary mb-2">Explainable</h3>
                <p className="text-sm font-medium text-muted-foreground">AI Clinical Summaries</p>
              </Card>
            </div>

            {/* ABOUT SECTION */}
            <div className="bg-secondary/30 border border-border rounded-3xl p-8 md:p-16 mb-20">
              <div className="text-center max-w-3xl mx-auto mb-12">
                <span className="text-xs uppercase tracking-widest text-primary font-bold">
                  About CardioAI
                </span>
                <h2 className="mt-4 text-2xl md:text-3xl lg:text-4xl font-extrabold tracking-tight text-foreground">
                  Next-Generation Cardiovascular Intelligence
                </h2>
                <p className="mt-6 text-sm md:text-base text-muted-foreground leading-relaxed max-w-2xl mx-auto">
                  CardioAI bridges modern diagnostic data with advanced artificial intelligence to give patients and physicians unprecedented clarity on cardiac health.
                </p>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
                {/* About 1: AI Prediction */}
                <Card className="p-8 hover:border-primary/40 transition-colors">
                  <div className="w-14 h-14 rounded-2xl bg-primary/10 text-primary flex items-center justify-center mb-6">
                    <Brain size={28} />
                  </div>
                  <h3 className="text-lg font-bold text-foreground mb-3">Multimodal AI</h3>
                  <p className="text-sm leading-relaxed text-muted-foreground">
                    Machine learning models evaluate clinical vitals, lab biomarkers, ECG waves, and echo imagery simultaneously.
                  </p>
                </Card>

                {/* About 2: Clinical Decision Support */}
                <Card className="p-8 hover:border-primary/40 transition-colors">
                  <div className="w-14 h-14 rounded-2xl bg-primary/10 text-primary flex items-center justify-center mb-6">
                    <ShieldCheck size={28} />
                  </div>
                  <h3 className="text-lg font-bold text-foreground mb-3">Decision Support</h3>
                  <p className="text-sm leading-relaxed text-muted-foreground">
                    Provides actionable risk stratification and confidence metrics to assist cardiologists in timely intervention.
                  </p>
                </Card>

                {/* About 3: Personalized Healthcare */}
                <Card className="p-8 hover:border-primary/40 transition-colors">
                  <div className="w-14 h-14 rounded-2xl bg-primary/10 text-primary flex items-center justify-center mb-6">
                    <UserCheck size={28} />
                  </div>
                  <h3 className="text-lg font-bold text-foreground mb-3">Personalized Care</h3>
                  <p className="text-sm leading-relaxed text-muted-foreground">
                    Tailors treatment guidance, lifestyle targets, and longitudinal tracking to each unique patient profile.
                  </p>
                </Card>

                {/* About 4: Digital Twin */}
                <Card className="p-8 hover:border-primary/40 transition-colors">
                  <div className="w-14 h-14 rounded-2xl bg-primary/10 text-primary flex items-center justify-center mb-6">
                    <Layers size={28} />
                  </div>
                  <h3 className="text-lg font-bold text-foreground mb-3">Digital Twin</h3>
                  <p className="text-sm leading-relaxed text-muted-foreground">
                    Interactive real-time cardiovascular simulation allowing users to model physiological parameter adjustments live.
                  </p>
                </Card>
              </div>
            </div>

            {/* WORKFLOW COMPONENT */}
            <div className="mb-20">
              <Workflow />
            </div>

            {/* FEATURES SECTION */}
            <div className="mb-20">
              <div className="text-center max-w-3xl mx-auto mb-12">
                <span className="text-xs uppercase tracking-widest text-primary font-bold">
                  Platform Capabilities
                </span>
                <h2 className="mt-4 text-2xl md:text-3xl lg:text-4xl font-extrabold tracking-tight text-foreground">
                  Comprehensive Cardiac Feature Suite
                </h2>
                <p className="mt-6 text-sm md:text-base text-muted-foreground leading-relaxed max-w-2xl mx-auto">
                  Designed for clarity, clinical rigor, and seamless interaction across every stage of cardiac care.
                </p>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
                {/* Card 1 */}
                <Card className="p-8 group hover:border-primary/40 transition-all hover:shadow-xl cursor-default">
                  <div className="w-14 h-14 rounded-2xl bg-primary/10 text-primary flex items-center justify-center mb-6 transition-transform group-hover:scale-110">
                    <Stethoscope size={28} />
                  </div>
                  <h3 className="text-lg font-bold text-foreground mb-3 group-hover:text-primary transition-colors">
                    Clinical Assessment
                  </h3>
                  <p className="text-sm leading-relaxed text-muted-foreground">
                    Evaluates vitals, blood pressure, cholesterol, fasting glucose, and lifestyle metrics for risk stratification.
                  </p>
                </Card>

                {/* Card 2 */}
                <Card className="p-8 group hover:border-primary/40 transition-all hover:shadow-xl cursor-default">
                  <div className="w-14 h-14 rounded-2xl bg-primary/10 text-primary flex items-center justify-center mb-6 transition-transform group-hover:scale-110">
                    <Activity size={28} />
                  </div>
                  <h3 className="text-lg font-bold text-foreground mb-3 group-hover:text-primary transition-colors">
                    ECG Waveform Analysis
                  </h3>
                  <p className="text-sm leading-relaxed text-muted-foreground">
                    Deep learning wave classification for automated detection of rhythmic and ischemic cardiac patterns.
                  </p>
                </Card>

                {/* Card 3 */}
                <Card className="p-8 group hover:border-primary/40 transition-all hover:shadow-xl cursor-default">
                  <div className="w-14 h-14 rounded-2xl bg-primary/10 text-primary flex items-center justify-center mb-6 transition-transform group-hover:scale-110">
                    <HeartPulse size={28} />
                  </div>
                  <h3 className="text-lg font-bold text-foreground mb-3 group-hover:text-primary transition-colors">
                    Echocardiogram Imaging
                  </h3>
                  <p className="text-sm leading-relaxed text-muted-foreground">
                    Evaluates ventricular wall motion and structural ejection metrics via AI image processing.
                  </p>
                </Card>

                {/* Card 4 */}
                <Card className="p-8 group hover:border-primary/40 transition-all hover:shadow-xl cursor-default">
                  <div className="w-14 h-14 rounded-2xl bg-primary/10 text-primary flex items-center justify-center mb-6 transition-transform group-hover:scale-110">
                    <Layers size={28} />
                  </div>
                  <h3 className="text-lg font-bold text-foreground mb-3 group-hover:text-primary transition-colors">
                    Digital Twin Simulation
                  </h3>
                  <p className="text-sm leading-relaxed text-muted-foreground">
                    Simulate 14 physiological parameter adjustments and visualize 10-year risk trajectories live.
                  </p>
                </Card>

                {/* Card 5 */}
                <Card className="p-8 group hover:border-primary/40 transition-all hover:shadow-xl cursor-default">
                  <div className="w-14 h-14 rounded-2xl bg-primary/10 text-primary flex items-center justify-center mb-6 transition-transform group-hover:scale-110">
                    <FileText size={28} />
                  </div>
                  <h3 className="text-lg font-bold text-foreground mb-3 group-hover:text-primary transition-colors">
                    Standardized Reports
                  </h3>
                  <p className="text-sm leading-relaxed text-muted-foreground">
                    Generate standardized, print-ready clinical summaries with detailed diagnostic findings.
                  </p>
                </Card>

                {/* Card 6 */}
                <Card className="p-8 group hover:border-primary/40 transition-all hover:shadow-xl cursor-default">
                  <div className="w-14 h-14 rounded-2xl bg-primary/10 text-primary flex items-center justify-center mb-6 transition-transform group-hover:scale-110">
                    <Calendar size={28} />
                  </div>
                  <h3 className="text-lg font-bold text-foreground mb-3 group-hover:text-primary transition-colors">
                    Specialist Consultation
                  </h3>
                  <p className="text-sm leading-relaxed text-muted-foreground">
                    Schedule consultations and share digital screening reports directly with certified cardiologists.
                  </p>
                </Card>
              </div>
            </div>

            {/* CTA BANNER */}
            <div className="mb-12">
              <Card className="p-10 md:p-16 text-center rounded-3xl bg-gradient-to-r from-primary/10 via-background to-primary/10 border-primary/20 shadow-lg">
                <h2 className="text-3xl md:text-4xl font-extrabold text-foreground tracking-tight">
                  Ready to Evaluate Cardiovascular Health?
                </h2>
                <p className="mt-6 text-base md:text-lg text-muted-foreground max-w-2xl mx-auto leading-relaxed">
                  Access the patient or clinician portal to experience CardioAI multimodal risk prediction.
                </p>
                <div className="mt-10 flex flex-wrap justify-center gap-4">
                  <Button
                    size="lg"
                    onClick={() => navigate("/get-started")}
                    className="gap-2 text-sm font-semibold h-12 px-8"
                  >
                    <span>Get Started Now</span>
                    <ArrowRight size={16} />
                  </Button>
                  <Button
                    variant="outline"
                    size="lg"
                    onClick={() => navigate("/")}
                    className="gap-2 text-sm font-semibold h-12 px-8"
                  >
                    <ArrowLeft size={16} />
                    <span>Return to Home</span>
                  </Button>
                </div>
              </Card>
            </div>
          </Container>
        </Section>
      </main>

      {/* FOOTER */}
      <footer className="py-8 border-t border-border text-sm text-muted-foreground bg-secondary/10">
        <Container className="flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2.5 font-bold text-foreground">
            <div className="w-8 h-8 rounded-xl bg-primary text-primary-foreground flex items-center justify-center text-xs font-black shadow-sm">
              AI
            </div>
            <span>CardioAI Intelligence</span>
          </div>

          <div className="text-center md:text-right">
            © {new Date().getFullYear()} CardioAI Platform. All rights reserved.
          </div>
        </Container>
      </footer>
    </AppLayout>
  );
}
