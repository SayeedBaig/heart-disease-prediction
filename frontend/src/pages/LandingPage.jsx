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
    <>
      {/* Navbar */}
      <Navbar />

      {/* Hero Section */}
      <Hero />

      {/* Workflow Section */}
      <Workflow />

      {/* AI Modules Section */}
      <section className="py-24 bg-gradient-to-b from-slate-50 to-white">

        <div className="max-w-7xl mx-auto px-6">

          {/* Section Heading */}

          <div className="text-center mb-16">

            <h2 className="text-4xl lg:text-5xl font-bold text-slate-900 mb-5">
              Multi-Modal AI Modules
            </h2>

            <p className="text-lg text-gray-600 max-w-3xl mx-auto leading-8">
              CardioAI combines multiple Artificial Intelligence models to
              provide accurate, explainable, and reliable cardiovascular
              risk prediction through a unified AI pipeline.
            </p>

          </div>

          {/* Cards */}

          <div className="grid lg:grid-cols-3 md:grid-cols-2 gap-10">

            <FeatureCard
              icon={Stethoscope}
              title="Clinical Analysis"
              description="Predict cardiovascular risk using patient demographics, blood pressure, cholesterol, glucose and lifestyle information."
            />

            <FeatureCard
              icon={Activity}
              title="ECG Analysis"
              description="Deep Learning based ECG analysis identifies cardiac abnormalities and improves prediction accuracy."
            />

            <FeatureCard
              icon={HeartPulse}
              title="Echo Analysis"
              description="AI-powered Echocardiography analysis evaluates cardiac function and strengthens multi-modal prediction."
            />

          </div>

        </div>

      </section>

      {/* Footer */}

      <Footer />
    </>
  );
}

export default LandingPage;