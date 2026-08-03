import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import {
  Activity,
  Image,
  History,
  Camera,
  FileDown,
  FileJson,
  Share2,
  Gauge,
} from "lucide-react";
import LiquidEther from "@/components/LiquidEther";
import ScrollStack, { ScrollStackItem } from "@/components/ScrollStack";

const mainCards = [
  {
    to: "/analysis",
    icon: Activity,
    title: "Analyze Movement",
    desc: "Upload trainer & user videos for AI biomechanical analysis",
    glow: true,
  },
  {
    to: "/generate",
    icon: Image,
    title: "Generate Correct Form",
    desc: "AI-powered form correction image generation",
    glow: true,
  },
  {
    to: "#",
    icon: History,
    title: "Past Analyses",
    desc: "Review your previous sessions and progress",
    glow: false,
  },
];

export default function Index() {
  return (
    <div className="min-h-screen relative bg-background">
      {/* --- BACKGROUND LAYER --- */}
      <div className="fixed inset-0 pointer-events-none w-full h-full z-0">
        <LiquidEther
          mouseForce={20}
          cursorSize={100}
          isViscous
          viscous={30}
          colors={["#29d4ff", "#a09eff", "#a3f0cf"]}
          autoDemo
          autoSpeed={0.5}
          autoIntensity={2.2}
          isBounce={false}
          resolution={0.5}
          className="w-full h-full"
        />
      </div>

      {/* --- CONTENT LAYER --- */}

      {/* Hero Section: Increased z-index to z-30 to stay above the negative margin overlap */}
      <section className="relative z-30 overflow-hidden pt-24 pb-0 md:pt-36 md:pb-6">
        <div className="container mx-auto px-4 relative">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center max-w-3xl mx-auto"
          >
            <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full glass text-xs font-medium text-primary mb-6">
              <Activity className="w-3.5 h-3.5" />
              AI-Powered Biomechanical Analysis
            </div>

            <h1 className="font-heading text-5xl md:text-7xl font-black leading-[0.95] mb-6">
              <span className="gradient-text">Visual AI</span>
              <br />
              Gym Coach
            </h1>

            <p className="text-lg md:text-xl text-muted-foreground mb-4 font-light">
              Upload. Analyze. Correct.
            </p>
            <p className="text-sm text-muted-foreground/70 max-w-lg mx-auto mb-12">
              Advanced computer vision for real-time exercise form analysis. Compare your technique against expert trainers and get instant AI feedback.
            </p>

            <div className="flex flex-wrap justify-center gap-3 relative z-40">
              <Link
                to="/analysis"
                className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-primary text-primary-foreground font-semibold hover:opacity-90 neon-glow transition-all cursor-pointer"
              >
                <Activity className="w-4 h-4" />
                Start Analysis
              </Link>
              <Link
                to="/generate"
                className="inline-flex items-center gap-2 px-6 py-3 rounded-xl glass text-foreground font-semibold hover:bg-secondary transition-all cursor-pointer"
              >
                <Image className="w-4 h-4" />
                Generate Image
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Scroll Stack Section: Kept at z-10 so the Hero stays on top */}
      <section className="relative z-10 w-full -mt-40">
        <ScrollStack>
          {mainCards.map((card) => (
            <ScrollStackItem className="h-[400px] max-w-4xl" key={card.title}>
              <Link to={card.to} className="group block w-full cursor-pointer">
                <div className="flex flex-col items-center gap-6">
                  <div
                    className={`w-24 h-24 rounded-3xl bg-primary/10 flex items-center justify-center group-hover:bg-primary/20 transition-all duration-500 ${
                      card.glow ? "neon-glow" : ""
                    }`}
                  >
                    <card.icon className="w-10 h-10 text-primary" />
                  </div>
                  <div>
                    <h2 className="font-heading text-3xl md:text-5xl font-bold text-foreground mb-4">
                      {card.title}
                    </h2>
                    <p className="text-lg text-muted-foreground max-w-md mx-auto leading-relaxed">
                      {card.desc}
                    </p>
                  </div>
                </div>
              </Link>
            </ScrollStackItem>
          ))}
        </ScrollStack>
      </section>

      <section className="container mx-auto px-4 pb-20 relative z-10">
        <div className="max-w-4xl mx-auto">
          {/* Quick features could go here */}
        </div>
      </section>
    </div>
  );
}