import Navbar from "../components/Navbar";
import Hero from "../components/Hero";
import Workflow from "../components/Workflow";
import FeatureCard from "../components/FeatureCard";
import Footer from "../components/Footer";

import {
  Stethoscope,
  Activity,
  HeartPulse,
} from "lucide-react";

function LandingPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-blue-50">

      {/* ================= Navbar ================= */}

      <Navbar />

      {/* ================= Hero ================= */}

      <Hero />

      {/* ================= Workflow ================= */}

      <section id="workflow">
        <Workflow />
      </section>

      {/* ================= AI MODULES ================= */}

      <section
        id="features"
        className="relative overflow-hidden py-28"
      >

        {/* Decorative Background */}

        <div className="absolute inset-0 bg-gradient-to-b from-blue-50/60 via-white to-slate-50"></div>

        {/* Decorative Blur */}

        <div className="absolute -top-16 left-10 w-72 h-72 bg-blue-200 rounded-full blur-3xl opacity-25"></div>

        <div className="absolute bottom-0 right-10 w-80 h-80 bg-cyan-200 rounded-full blur-3xl opacity-20"></div>

        <div className="relative max-w-7xl mx-auto px-6">

          {/* Badge */}

          <div className="flex justify-center mb-6">

            <span className="px-5 py-2 rounded-full bg-blue-100 text-blue-700 text-sm font-semibold tracking-wide shadow-sm">
              AI Powered Healthcare Platform
            </span>

          </div>

          {/* Heading */}

          <div className="text-center max-w-4xl mx-auto">

            <h2 className="text-4xl md:text-5xl lg:text-6xl font-bold tracking-tight text-slate-900 leading-tight">

              Powered by{" "}

              <span className="text-blue-600">
                Advanced AI Models
              </span>

            </h2>

            <p className="mt-8 text-lg md:text-xl leading-8 text-slate-600">

              CardioAI integrates Clinical Analysis,
              ECG Deep Learning, and Echocardiography
              Intelligence into a single intelligent
              platform for accurate, explainable,
              and reliable cardiovascular disease prediction.

            </p>

          </div>

          {/* Feature Cards */}

          <div className="mt-20 grid gap-10 md:grid-cols-2 lg:grid-cols-3">

            <FeatureCard
              icon={Stethoscope}
              title="Clinical Analysis"
              description="Predict cardiovascular risk using patient demographics, blood pressure, cholesterol, glucose levels, and lifestyle information."
            />

            <FeatureCard
              icon={Activity}
              title="ECG Analysis"
              description="Deep Learning powered ECG interpretation detects cardiac abnormalities and significantly improves prediction accuracy."
            />

            <FeatureCard
              icon={HeartPulse}
              title="Echo Analysis"
              description="AI-driven Echocardiography analysis evaluates cardiac structure and function for comprehensive multi-modal diagnosis."
            />

          </div>

        </div>

      </section>

      {/* ================= Footer ================= */}

      <section id="about">
        <Footer />
      </section>

    </div>
  );
}

export default LandingPage;