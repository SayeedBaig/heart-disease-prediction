import Navbar from "../components/Navbar";
import Hero from "../components/Hero";
import FeatureCard from "../components/FeatureCard";
import Footer from "../components/Footer";

function LandingPage() {
  return (
    <>
      <Navbar />

      <Hero />

      <section className="bg-gray-100 py-20">
        <h2 className="text-4xl font-bold text-center text-blue-900 mb-12">
          Our AI Modules
        </h2>

        <div className="grid md:grid-cols-3 gap-10 max-w-7xl mx-auto px-6">

          <FeatureCard
            icon="🩺"
            title="Clinical Analysis"
            description="Predict heart disease using patient history and clinical parameters."
          />

          <FeatureCard
            icon="📈"
            title="ECG Analysis"
            description="Deep Learning based ECG signal analysis for cardiovascular prediction."
          />

          <FeatureCard
            icon="🫀"
            title="Echo Analysis"
            description="Analyze Echocardiography videos to estimate cardiac function."
          />

        </div>
      </section>

      <Footer />
    </>
  );
}

export default LandingPage;