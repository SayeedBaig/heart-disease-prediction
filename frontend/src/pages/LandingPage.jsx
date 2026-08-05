import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { 
  ChevronDown, ArrowRight, CheckCircle2, Sparkles, UserPlus, 
  Activity, HeartPulse, ClipboardList, BrainCircuit, FileText,
  MessageCircle, Send, X, Bot, ShieldCheck, Play, LineChart, 
  Settings, UserCheck, Stethoscope
} from "lucide-react";
import { AppLayout } from "../components/ui/AppLayout";
import { Container } from "../components/ui/Container";
import { Section } from "../components/ui/Section";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "../components/ui/Card";
import { Button } from "../components/ui/Button";
import { Grid } from "../components/ui/Grid";

import { stats, capabilities, reasons, faqs, twinMetrics } from "../components/landing/landingData";

const workflowSteps = [
  { icon: UserPlus, title: "Patient Intake", description: "Capture demographic vitals and clinical history." },
  { icon: ClipboardList, title: "Clinical Vitals", description: "Input biomarkers, lipids, and glucose metrics." },
  { icon: Activity, title: "ECG Waveforms", description: "Automated neural network wave classification." },
  { icon: HeartPulse, title: "Echo Imaging", description: "AI evaluates ejection fraction and wall motion." },
  { icon: BrainCircuit, title: "Multimodal Fusion", description: "Synthesize all data into a unified risk vector." },
  { icon: FileText, title: "Clinical Report", description: "Generate standardized specialist summaries." },
];

export default function LandingPage() {
  const scrollToCapabilities = () => {
    document.getElementById("capabilities")?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <AppLayout className="bg-background overflow-hidden selection:bg-primary/20">
      
      {/* ================= HERO SECTION ================= */}
      <Section className="relative pt-20 md:pt-32 pb-24 md:pb-40 border-b border-border z-10">
        {/* Background Glow Effects */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[600px] bg-primary/10 rounded-full blur-[120px] opacity-70 pointer-events-none -z-10" />
        <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-blue-500/10 rounded-full blur-[100px] opacity-50 pointer-events-none -z-10" />

        <Container>
          <Grid cols={2} gap={12} className="items-center">
            {/* Left Content */}
            <motion.div 
              initial={{ opacity: 0, y: 20 }} 
              animate={{ opacity: 1, y: 0 }} 
              transition={{ duration: 0.6 }}
            >
              <div className="inline-flex items-center gap-2 rounded-full border border-primary/20 bg-primary/5 px-4 py-1.5 text-xs font-semibold text-primary shadow-sm backdrop-blur-sm cursor-default">
                <Sparkles className="h-4 w-4" />
                Next-Generation Medical AI Platform
              </div>
              <h1 className="mt-8 text-5xl font-extrabold tracking-tight sm:text-6xl lg:text-7xl text-foreground leading-[1.1]">
                AI-Powered Heart <br className="hidden lg:block"/>
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary to-blue-500">
                  Disease Screening
                </span>
              </h1>
              <p className="mt-6 text-lg md:text-xl text-muted-foreground max-w-xl leading-relaxed">
                Easily understand your heart health with our advanced AI. We analyze your vitals, ECG, and echocardiogram results to give you a clear and instant risk report.
              </p>
              <div className="mt-10 flex flex-wrap gap-4 items-center">
                <Link to="/get-started">
                  <Button size="lg" className="h-14 px-8 text-base font-semibold shadow-lg shadow-primary/20 group transition-all hover:-translate-y-0.5 cursor-pointer">
                    Start Screening
                    <ArrowRight className="ml-2 h-5 w-5 group-hover:translate-x-1 transition-transform" />
                  </Button>
                </Link>
                <Button variant="outline" size="lg" onClick={scrollToCapabilities} className="h-14 px-6 text-base font-semibold bg-background/50 backdrop-blur hover:bg-accent transition-all hover:-translate-y-0.5 cursor-pointer">
                  Explore Capabilities
                  <ChevronDown className="ml-2 h-5 w-5" />
                </Button>
              </div>
              
              <div className="mt-12 flex items-center gap-4 text-sm text-muted-foreground font-medium">
                <div className="flex -space-x-2">
                  {[1,2,3,4].map(i => (
                    <div key={i} className="w-8 h-8 rounded-full border-2 border-background bg-secondary flex items-center justify-center">
                      <UserCheck className="h-4 w-4 text-primary" />
                    </div>
                  ))}
                </div>
                <div>Secure & Private Health Platform</div>
              </div>
            </motion.div>

            {/* Right Content - Interactive AI Mock */}
            <motion.div 
              initial={{ opacity: 0, scale: 0.95 }} 
              animate={{ opacity: 1, scale: 1 }} 
              transition={{ duration: 0.8, delay: 0.2 }}
              className="relative w-full max-w-[540px] mx-auto lg:ml-auto perspective-1000 cursor-default"
            >
              {/* Floating element 1 */}
              <motion.div 
                animate={{ y: [0, -10, 0] }} 
                transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
                className="absolute -top-6 -right-6 z-20 bg-background border border-border shadow-xl rounded-xl p-4 flex items-center gap-3 backdrop-blur-md"
              >
                <div className="h-10 w-10 bg-success/10 rounded-full flex items-center justify-center">
                  <Activity className="h-5 w-5 text-success" />
                </div>
                <div>
                  <div className="text-xs font-bold text-muted-foreground uppercase">ECG Analysis</div>
                  <div className="text-sm font-extrabold">Normal Sinus Rhythm</div>
                </div>
              </motion.div>

              {/* Main Dashboard Panel */}
              <div className="relative rounded-2xl border border-slate-800 bg-slate-950 shadow-2xl overflow-hidden group">
                <div className="h-1.5 w-full bg-gradient-to-r from-primary via-blue-400 to-indigo-500" />
                <div className="p-6">
                  <div className="flex justify-between items-center mb-6">
                    <div>
                      <div className="text-xs font-bold uppercase tracking-wider text-blue-400 mb-1">Live AI Evaluation</div>
                      <h3 className="text-xl font-extrabold text-white">Patient Risk Profile</h3>
                    </div>
                    <div className="flex items-center gap-2 text-[10px] font-bold uppercase px-2.5 py-1 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                      <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                      Analyzing
                    </div>
                  </div>

                  {/* Animated ECG Line */}
                  <div className="h-24 w-full bg-slate-900 rounded-xl border border-slate-800 flex items-center justify-center mb-6 overflow-hidden relative">
                    <svg viewBox="0 0 400 100" className="w-full h-full opacity-60 stroke-blue-500" fill="none" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                      <motion.path 
                        d="M 0 50 L 50 50 L 70 20 L 90 80 L 110 50 L 180 50 L 200 10 L 220 90 L 240 50 L 320 50 L 340 30 L 360 70 L 380 50 L 450 50"
                        initial={{ pathLength: 0, x: 0 }}
                        animate={{ pathLength: 1, x: -50 }}
                        transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                      />
                    </svg>
                    <div className="absolute inset-0 bg-gradient-to-r from-slate-950 via-transparent to-slate-950" />
                  </div>

                  {/* Metrics Grid */}
                  <div className="grid grid-cols-2 gap-4 mb-6">
                    <div className="p-4 rounded-xl border border-slate-800 bg-slate-900">
                      <div className="text-xs text-slate-400 font-medium mb-1">AI Confidence Score</div>
                      <div className="text-2xl font-extrabold text-emerald-400">96.4%</div>
                    </div>
                    <div className="p-4 rounded-xl border border-slate-800 bg-slate-900">
                      <div className="text-xs text-slate-400 font-medium mb-1">Calculated Risk</div>
                      <div className="text-2xl font-extrabold text-amber-400">Moderate</div>
                    </div>
                  </div>

                  <div className="space-y-3">
                    <div className="flex items-center gap-3 p-3 rounded-lg bg-slate-900 border border-slate-800 text-slate-200">
                      <CheckCircle2 className="h-5 w-5 text-emerald-500 shrink-0" />
                      <div className="text-sm font-medium">Ejection Fraction evaluated at 58%</div>
                    </div>
                    <div className="flex items-center gap-3 p-3 rounded-lg bg-slate-900 border border-slate-800 text-slate-200">
                      <CheckCircle2 className="h-5 w-5 text-emerald-500 shrink-0" />
                      <div className="text-sm font-medium">Lipid panels within nominal ranges</div>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          </Grid>
        </Container>
      </Section>
      
      {/* ================= STATS SECTION ================= */}
      <Section className="bg-secondary border-b border-border">
        <Container>
          <Grid cols={4} gap={6}>
            {stats.map((s, i) => (
              <motion.div 
                key={s.label}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1, duration: 0.5 }}
              >
                <Card className="text-center hover:shadow-lg hover:border-primary/40 transition-all duration-300 h-full bg-background group cursor-pointer">
                  <CardContent className="pt-8 pb-8">
                    <div className="text-4xl md:text-5xl font-extrabold text-primary group-hover:scale-105 transition-transform">{s.value}</div>
                    <div className="mt-4 font-bold text-sm text-foreground uppercase tracking-wide">{s.label}</div>
                    <div className="mt-2 text-xs text-muted-foreground leading-relaxed">{s.desc}</div>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </Grid>
        </Container>
      </Section>

      {/* ================= CAPABILITIES SECTION ================= */}
      <Section id="capabilities" className="bg-background relative">
        <Container>
          <div className="text-center max-w-3xl mx-auto mb-16">
            <span className="inline-block py-1 px-3 rounded-full bg-primary/10 text-primary text-xs font-bold uppercase tracking-widest mb-4 border border-primary/20">
              Multimodal Diagnostics
            </span>
            <h2 className="text-3xl font-extrabold tracking-tight sm:text-4xl lg:text-5xl text-foreground">
              Comprehensive Cardiac Intelligence
            </h2>
            <p className="mt-4 text-muted-foreground text-lg leading-relaxed">
              Designed for clinical accuracy, seamless data integration, and intuitive diagnostic visualization across every step of cardiovascular care.
            </p>
          </div>
          
          <Grid cols={3} gap={8}>
            {capabilities.map((c, i) => {
              const Icon = c.icon;
              return (
                <motion.div 
                  key={c.title}
                  whileHover={{ y: -8 }}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: i * 0.1, duration: 0.4 }}
                  className="h-full"
                >
                  <Card className="h-full hover:shadow-xl hover:border-primary/50 transition-all duration-300 border-border bg-background group relative overflow-hidden cursor-pointer">
                    <div className="absolute top-0 right-0 p-16 bg-primary/5 rounded-full blur-3xl -mr-10 -mt-10 group-hover:bg-primary/10 transition-colors" />
                    <CardHeader className="relative z-10">
                      <div className="h-14 w-14 rounded-2xl bg-primary/10 flex items-center justify-center mb-5 border border-primary/20 group-hover:bg-primary group-hover:text-primary-foreground transition-colors shadow-sm">
                        <Icon className="h-7 w-7 text-primary group-hover:text-primary-foreground transition-colors" />
                      </div>
                      <CardTitle className="text-xl font-bold">{c.title}</CardTitle>
                    </CardHeader>
                    <CardContent className="relative z-10">
                      <CardDescription className="text-sm leading-relaxed text-muted-foreground group-hover:text-foreground/80 transition-colors">
                        {c.description}
                      </CardDescription>
                    </CardContent>
                  </Card>
                </motion.div>
              );
            })}
          </Grid>
        </Container>
      </Section>

      {/* ================= WORKFLOW SECTION ================= */}
      <Section className="bg-secondary border-y border-border">
        <Container>
          <div className="text-center max-w-3xl mx-auto mb-16">
            <h2 className="text-3xl font-extrabold tracking-tight sm:text-4xl text-foreground">
              Seamless Clinical Workflow
            </h2>
            <p className="mt-4 text-muted-foreground text-lg leading-relaxed">
              A streamlined, end-to-end flow from patient intake to comprehensive AI-generated diagnostic reporting.
            </p>
          </div>
          
          <Grid cols={3} gap={6}>
            {workflowSteps.map((s, i) => {
              const Icon = s.icon;
              return (
                <motion.div
                  key={s.title}
                  initial={{ opacity: 0, scale: 0.95 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  viewport={{ once: true }}
                  transition={{ delay: i * 0.05, duration: 0.4 }}
                  className="h-full"
                >
                  <Card className="h-full border-border bg-background relative hover:shadow-lg transition-shadow cursor-pointer">
                    <CardHeader>
                      <div className="flex items-center justify-between mb-4">
                        <div className="h-12 w-12 rounded-xl bg-secondary flex items-center justify-center border border-border">
                          <Icon className="h-6 w-6 text-foreground" />
                        </div>
                        <div className="text-4xl font-black text-secondary-foreground/10 italic">
                          0{i + 1}
                        </div>
                      </div>
                      <CardTitle className="text-lg font-bold">{s.title}</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <CardDescription className="leading-relaxed">
                        {s.description}
                      </CardDescription>
                    </CardContent>
                  </Card>
                </motion.div>
              );
            })}
          </Grid>
        </Container>
      </Section>

      {/* ================= DIGITAL TWIN SECTION ================= */}
      <Section className="bg-background border-b border-border overflow-hidden">
        <Container>
          <Grid cols={2} gap={8} className="items-center">
            <motion.div 
              initial={{ opacity: 0, x: -30 }} 
              whileInView={{ opacity: 1, x: 0 }} 
              viewport={{ once: true }}
              transition={{ duration: 0.6 }}
            >
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-500/10 text-blue-500 text-xs font-bold border border-blue-500/20 mb-6 cursor-default">
                <LineChart className="h-4 w-4" />
                Interactive Simulation Engine
              </div>
              <h2 className="text-3xl font-extrabold tracking-tight sm:text-4xl lg:text-5xl text-foreground leading-tight">
                Simulate future risk before it becomes critical
              </h2>
              <p className="mt-6 text-lg text-muted-foreground leading-relaxed">
                Our physiological Digital Twin models how targeted risk interventions—like blood pressure management or lifestyle adjustments—impact a patient's 10-year cardiac trajectory dynamically.
              </p>
              <div className="mt-8">
                <Link to="/get-started">
                  <Button size="lg" className="h-12 px-6 font-semibold shadow-md hover:-translate-y-0.5 transition-transform cursor-pointer">
                    <Play className="mr-2 h-4 w-4 fill-current" />
                    Launch Digital Twin Demo
                  </Button>
                </Link>
              </div>
            </motion.div>

            <motion.div 
              initial={{ opacity: 0, x: 30 }} 
              whileInView={{ opacity: 1, x: 0 }} 
              viewport={{ once: true }}
              transition={{ duration: 0.6 }}
              className="relative w-full cursor-default"
            >
              <div className="absolute inset-0 bg-blue-500/10 blur-[80px] rounded-full pointer-events-none" />
              <Card className="shadow-2xl border-border bg-background/80 backdrop-blur-xl relative z-10 hover:border-blue-500/30 transition-colors">
                <CardHeader className="border-b border-border pb-5 bg-secondary/20">
                  <div className="flex justify-between items-center">
                    <div>
                      <div className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground mb-1">Parameter Sandbox</div>
                      <CardTitle className="text-xl font-bold flex items-center gap-2">
                        10-Year Risk Trajectory
                      </CardTitle>
                    </div>
                    <div className="h-10 w-10 bg-blue-500/10 rounded-xl flex items-center justify-center border border-blue-500/20">
                      <Settings className="h-5 w-5 text-blue-500 animate-[spin_4s_linear_infinite]" />
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="space-y-6 pt-6">
                  {twinMetrics.map((m, i) => (
                    <motion.div 
                      key={m.label} 
                      className="space-y-2 group cursor-pointer"
                      whileHover={{ scale: 1.02 }}
                      transition={{ type: "spring", stiffness: 400, damping: 25 }}
                    >
                      <div className="flex justify-between text-xs font-bold">
                        <span className="text-foreground">{m.label}</span>
                        <span className="text-blue-500 bg-blue-500/10 px-2 py-0.5 rounded-md">{m.value}</span>
                      </div>
                      <div className="h-3 w-full bg-secondary rounded-full overflow-hidden border border-border relative">
                        <motion.div 
                          className="absolute top-0 left-0 h-full bg-blue-500 rounded-full" 
                          initial={{ width: 0 }}
                          whileInView={{ width: m.percent }}
                          viewport={{ once: true }}
                          transition={{ duration: 1, delay: 0.2 + (i * 0.1), ease: "easeOut" }}
                        />
                      </div>
                    </motion.div>
                  ))}
                </CardContent>
              </Card>
            </motion.div>
          </Grid>
        </Container>
      </Section>

      {/* ================= REPORT PREVIEW SECTION ================= */}
      <Section className="bg-secondary border-b border-border">
        <Container>
          <Grid cols={2} gap={8} className="items-center">
            <motion.div 
              className="order-2 md:order-1 relative w-full cursor-pointer"
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6 }}
            >
              <Card className="shadow-2xl border-border bg-background group hover:border-primary/30 transition-colors">
                <CardHeader className="border-b border-border pb-5 bg-secondary/10">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className="h-10 w-10 bg-primary rounded-xl flex items-center justify-center text-primary-foreground font-black shadow-md">
                        AI
                      </div>
                      <div>
                        <CardTitle className="text-lg font-bold">CardioAI Official Report</CardTitle>
                        <CardDescription className="text-xs">Multimodal Diagnostic Summary</CardDescription>
                      </div>
                    </div>
                    <div className="px-2 py-1 bg-success/10 text-success text-[10px] font-bold uppercase rounded-md border border-success/20">
                      Verified
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="pt-6">
                  <Grid cols={3} gap={4} className="mb-6">
                    <div className="bg-background rounded-xl p-3 text-center border border-border shadow-sm group-hover:border-primary/20 transition-colors">
                      <div className="text-[10px] font-bold uppercase text-muted-foreground">Risk Level</div>
                      <div className="text-lg font-extrabold mt-1 text-warning">Moderate</div>
                    </div>
                    <div className="bg-background rounded-xl p-3 text-center border border-border shadow-sm group-hover:border-primary/20 transition-colors">
                      <div className="text-[10px] font-bold uppercase text-muted-foreground">AI Confidence</div>
                      <div className="text-lg font-extrabold mt-1 text-success">96.4%</div>
                    </div>
                    <div className="bg-background rounded-xl p-3 text-center border border-border shadow-sm group-hover:border-primary/20 transition-colors">
                      <div className="text-[10px] font-bold uppercase text-muted-foreground">ECG Pattern</div>
                      <div className="text-lg font-extrabold mt-1 text-foreground">Normal</div>
                    </div>
                  </Grid>
                  
                  <div className="space-y-4">
                    <div className="text-xs text-muted-foreground leading-relaxed border-l-2 border-primary/50 pl-4 py-1">
                      <span className="font-bold text-foreground">Clinical Note:</span> Screening findings indicate a low-to-moderate cardiovascular risk trajectory. Routine monitoring and preventive lifestyle modifications (dietary adjustments) are recommended. No immediate ischemic interventions required.
                    </div>
                  </div>
                  
                  <div className="mt-6 flex items-center justify-between pt-4 border-t border-border">
                    <div className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
                      <Stethoscope className="h-4 w-4" />
                      Electronically signed by CardioAI
                    </div>
                    <Button variant="ghost" size="sm" className="h-8 text-xs font-semibold cursor-pointer">
                      View Full PDF
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </motion.div>

            <motion.div 
              className="order-1 md:order-2"
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6 }}
            >
              <h2 className="text-3xl font-extrabold tracking-tight sm:text-4xl text-foreground leading-tight">
                Explainable clinical reports designed for clarity
              </h2>
              <p className="mt-5 text-lg text-muted-foreground leading-relaxed">
                Generate standardized, print-ready summaries with transparent risk breakdowns, confidence metrics, automated ECG interpretations, and physician notes ready for EHR export.
              </p>
              <div className="mt-8 flex gap-4">
                <Button variant="outline" size="lg" className="h-12 font-semibold cursor-pointer">
                  <FileText className="mr-2 h-4 w-4" />
                  View Sample Report
                </Button>
              </div>
            </motion.div>
          </Grid>
        </Container>
      </Section>

      {/* ================= WHY CHOOSE SECTION ================= */}
      <Section className="bg-background border-b border-border">
        <Container>
          <div className="text-center max-w-3xl mx-auto mb-16">
            <h2 className="text-3xl font-extrabold tracking-tight sm:text-4xl text-foreground">
              Why Choose CardioAI
            </h2>
            <p className="mt-4 text-muted-foreground text-lg leading-relaxed">
              Bridging diagnostic medical data with modern artificial intelligence to deliver unprecedented clarity for healthcare providers.
            </p>
          </div>
          
          <Grid cols={3} gap={6}>
            {reasons.map((r, i) => {
              const Icon = r.icon;
              return (
                <motion.div
                  key={r.title}
                  whileHover={{ y: -8 }}
                  initial={{ opacity: 0, scale: 0.95 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  viewport={{ once: true }}
                  transition={{ delay: i * 0.1, duration: 0.4 }}
                  className="h-full"
                >
                  <Card className="h-full hover:shadow-xl hover:border-primary/50 transition-all duration-300 border-border bg-background group cursor-pointer">
                    <CardContent className="pt-8">
                      <div className="h-14 w-14 rounded-2xl bg-primary/10 flex items-center justify-center mb-6 border border-primary/20 group-hover:scale-110 group-hover:bg-primary transition-all shadow-sm">
                        <Icon className="h-7 w-7 text-primary group-hover:text-primary-foreground transition-colors" />
                      </div>
                      <CardTitle className="text-xl font-bold mb-3">{r.title}</CardTitle>
                      <CardDescription className="text-sm leading-relaxed text-muted-foreground group-hover:text-foreground/80 transition-colors">
                        {r.description}
                      </CardDescription>
                    </CardContent>
                  </Card>
                </motion.div>
              );
            })}
          </Grid>
        </Container>
      </Section>

      {/* ================= FAQ SECTION ================= */}
      <Section className="bg-secondary border-b border-border">
        <Container>
          <div className="text-center max-w-2xl mx-auto mb-12">
            <h2 className="text-3xl font-extrabold tracking-tight text-foreground">
              Frequently Asked Questions
            </h2>
          </div>
          <div className="max-w-3xl mx-auto space-y-4">
            {faqs.map((faq, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 10 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.05, duration: 0.4 }}
              >
                <FaqItem question={faq.question} answer={faq.answer} />
              </motion.div>
            ))}
          </div>
        </Container>
      </Section>

      {/* ================= FINAL CTA ================= */}
      <Section className="bg-background relative">
        <Container>
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
          >
            <Card className="bg-primary text-primary-foreground border-none overflow-hidden relative shadow-2xl">
              <div className="absolute top-0 right-0 p-40 opacity-20 blur-3xl rounded-full bg-background pointer-events-none" />
              <div className="absolute bottom-0 left-0 p-32 opacity-10 blur-3xl rounded-full bg-background pointer-events-none" />
              <CardContent className="flex flex-col items-center text-center p-16 md:p-24 space-y-6 relative z-10">
                <h2 className="text-4xl font-extrabold tracking-tight sm:text-5xl max-w-3xl leading-tight">
                  Ready to elevate cardiovascular screening?
                </h2>
                <p className="text-lg opacity-90 max-w-2xl leading-relaxed">
                  Join hundreds of medical practitioners using multimodal AI heart disease screening, explainable diagnostic reports, and interactive Digital Twin simulations.
                </p>
                <div className="mt-10">
                  <Link to="/get-started">
                    <Button variant="secondary" size="lg" className="h-14 px-8 font-bold text-primary shadow-xl hover:shadow-2xl hover:scale-105 transition-all duration-300 cursor-pointer">
                        Get Started Now <ArrowRight className="ml-2 h-5 w-5" />
                    </Button>
                  </Link>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        </Container>
      </Section>

      {/* ================= FOOTER ================= */}
      <footer className="border-t border-border bg-background py-16 text-sm text-muted-foreground relative z-10">
        <Container>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-12">
            <div className="md:col-span-2">
                <div className="flex items-center gap-3 font-extrabold text-foreground text-lg">
                  <div className="h-10 w-10 bg-primary rounded-xl text-primary-foreground flex items-center justify-center text-sm font-black shadow-md border border-primary/20">AI</div>
                  <span>CardioAI Intelligence</span>
                </div>
                <p className="mt-6 max-w-sm leading-relaxed text-sm">
                  Next-generation cardiovascular screening, explainable AI reports, and Digital Twin simulation built for modern clinical decision support.
                </p>
            </div>
            <div>
                <h3 className="font-bold text-foreground uppercase tracking-widest text-xs mb-6">Platform</h3>
                <div className="flex flex-col space-y-4 font-medium">
                  <button onClick={scrollToCapabilities} className="hover:text-primary transition-colors text-left w-fit cursor-pointer">Capabilities</button>
                  <Link to="/get-started" className="hover:text-primary transition-colors w-fit cursor-pointer">Clinical Workflow</Link>
                  <Link to="/get-started" className="hover:text-primary transition-colors w-fit cursor-pointer">Digital Twin</Link>
                  <Link to="/get-started" className="hover:text-primary transition-colors w-fit cursor-pointer">Access Portals</Link>
                </div>
            </div>
            <div>
                <h3 className="font-bold text-foreground uppercase tracking-widest text-xs mb-6">Medical Disclaimer</h3>
                <p className="leading-relaxed opacity-80 text-xs">
                  Designed strictly for clinical decision support. The platform is not a replacement for formal physician diagnosis, emergency services, or professional medical advice.
                </p>
            </div>
          </div>
          <div className="mt-16 pt-8 border-t border-border flex flex-col md:flex-row justify-between items-center gap-4 text-xs font-medium opacity-80">
            <div>© {new Date().getFullYear()} CardioAI Platform. All rights reserved.</div>
            <div className="flex gap-6">
              <a href="#" className="hover:text-foreground transition-colors cursor-pointer">Privacy Policy</a>
              <a href="#" className="hover:text-foreground transition-colors cursor-pointer">Terms of Service</a>
            </div>
          </div>
        </Container>
      </footer>

      {/* Floating Mock Assistant */}
      <MockFloatingAssistant />

    </AppLayout>
  );
}

function FaqItem({ question, answer }) {
  const [isOpen, setIsOpen] = useState(false);
  return (
    <Card 
      className={`cursor-pointer transition-all duration-300 ${isOpen ? 'border-primary shadow-md bg-background' : 'hover:border-primary/50 bg-background/50'}`}
      noPadding 
    >
      <div 
        className="p-6 flex justify-between items-center select-none" 
        onClick={() => setIsOpen(!isOpen)}
        role="button"
        tabIndex={0}
      >
        <span className="font-bold text-foreground pr-8 text-base">{question}</span>
        <div className={`h-8 w-8 rounded-lg flex items-center justify-center shrink-0 transition-colors duration-300 ${isOpen ? 'bg-primary text-primary-foreground' : 'bg-secondary text-muted-foreground'}`}>
          <ChevronDown className={`h-5 w-5 transition-transform duration-300 ${isOpen ? "rotate-180" : ""}`} />
        </div>
      </div>
      <AnimatePresence>
        {isOpen && (
          <motion.div 
            initial={{ height: 0, opacity: 0 }} 
            animate={{ height: "auto", opacity: 1 }} 
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: "easeInOut" }}
            className="overflow-hidden"
          >
            <div className="px-6 pb-6 pt-2 text-muted-foreground text-sm leading-relaxed border-t border-border mt-2">
              {answer}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </Card>
  );
}

function MockFloatingAssistant() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);

  // Initialize with a welcome message when opened for the first time
  useEffect(() => {
    if (isOpen && messages.length === 0) {
      setMessages([
        { id: 1, sender: "ai", text: "Hello! I am the CardioAI Assistant. I can help explain risk metrics, guidelines, or navigate the platform. How can I help you today?" }
      ]);
    }
  }, [isOpen, messages.length]);

  const handleSend = () => {
    if (!input.trim()) return;
    
    const userMsg = { id: Date.now(), sender: "user", text: input.trim() };
    setMessages(prev => [...prev, userMsg]);
    setInput("");
    setIsTyping(true);

    setTimeout(() => {
      const aiMsg = { 
        id: Date.now() + 1, 
        sender: "ai", 
        text: "This is a frontend demonstration of the CardioAI RAG Assistant. In production, this securely connects to our LLM for contextual clinical support." 
      };
      setMessages(prev => [...prev, aiMsg]);
      setIsTyping(false);
    }, 1500);
  };

  return (
    <div className="fixed bottom-6 right-6 z-50 flex flex-col items-end">
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: 20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 20, scale: 0.95 }}
            transition={{ duration: 0.2, type: "spring", damping: 25 }}
            className="mb-4 w-[360px] h-[500px] bg-background border border-border rounded-2xl shadow-2xl flex flex-col overflow-hidden"
          >
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b border-border bg-secondary/30 backdrop-blur">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-full bg-primary/10 text-primary flex items-center justify-center border border-primary/20">
                  <Bot size={18} />
                </div>
                <div>
                  <div className="text-sm font-bold text-foreground leading-none">AI Health Assistant</div>
                  <div className="text-[10px] font-medium text-success mt-1 flex items-center gap-1">
                    <div className="w-1.5 h-1.5 rounded-full bg-success animate-pulse" /> Online
                  </div>
                </div>
              </div>
              <button 
                onClick={() => setIsOpen(false)}
                className="h-8 w-8 rounded-full flex items-center justify-center text-muted-foreground hover:bg-secondary hover:text-foreground transition-colors cursor-pointer"
              >
                <X size={18} />
              </button>
            </div>

            {/* Chat Area */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-secondary/5">
              {messages.map(m => (
                <motion.div 
                  key={m.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`flex ${m.sender === "user" ? "justify-end" : "justify-start"}`}
                >
                  <div className={`max-w-[85%] p-3 rounded-2xl text-sm leading-relaxed shadow-sm ${
                    m.sender === "user" 
                      ? "bg-primary text-primary-foreground rounded-br-none" 
                      : "bg-background border border-border text-foreground rounded-bl-none"
                  }`}>
                    {m.text}
                  </div>
                </motion.div>
              ))}
              {isTyping && (
                <motion.div 
                  initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                  className="flex justify-start"
                >
                  <div className="bg-background border border-border rounded-2xl rounded-bl-none px-4 py-3 flex gap-1 items-center h-10 shadow-sm">
                    <div className="w-1.5 h-1.5 rounded-full bg-muted-foreground animate-bounce" style={{ animationDelay: '0ms' }} />
                    <div className="w-1.5 h-1.5 rounded-full bg-muted-foreground animate-bounce" style={{ animationDelay: '150ms' }} />
                    <div className="w-1.5 h-1.5 rounded-full bg-muted-foreground animate-bounce" style={{ animationDelay: '300ms' }} />
                  </div>
                </motion.div>
              )}
            </div>

            {/* Input Area */}
            <div className="p-3 bg-background border-t border-border">
              <div className="flex items-center gap-2">
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleSend()}
                  placeholder="Ask a medical question..."
                  className="flex-1 h-10 rounded-full border border-border bg-secondary/30 px-4 text-sm focus:outline-none focus:ring-1 focus:ring-primary transition-shadow placeholder:text-muted-foreground"
                />
                <button 
                  onClick={handleSend}
                  disabled={!input.trim()}
                  className="h-10 w-10 rounded-full bg-primary text-primary-foreground flex items-center justify-center shrink-0 disabled:opacity-50 disabled:cursor-not-allowed hover:bg-primary/90 transition-colors shadow-sm cursor-pointer"
                >
                  <Send size={16} className="ml-0.5" />
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <motion.button
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        onClick={() => setIsOpen(!isOpen)}
        className="h-14 w-14 rounded-full bg-primary text-primary-foreground shadow-xl flex items-center justify-center border border-primary/20 relative z-50 group cursor-pointer"
      >
        <AnimatePresence mode="wait">
          {isOpen ? (
            <motion.div key="close" initial={{ opacity: 0, rotate: -90 }} animate={{ opacity: 1, rotate: 0 }} exit={{ opacity: 0, rotate: 90 }} transition={{ duration: 0.2 }}>
              <X size={24} />
            </motion.div>
          ) : (
            <motion.div key="chat" initial={{ opacity: 0, rotate: 90 }} animate={{ opacity: 1, rotate: 0 }} exit={{ opacity: 0, rotate: -90 }} transition={{ duration: 0.2 }}>
              <MessageCircle size={24} />
            </motion.div>
          )}
        </AnimatePresence>
        {!isOpen && (
          <span className="absolute -top-1 -right-1 flex h-4 w-4">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-destructive opacity-75"></span>
            <span className="relative inline-flex rounded-full h-4 w-4 bg-destructive border-2 border-primary border-solid"></span>
          </span>
        )}
      </motion.button>
    </div>
  );
}