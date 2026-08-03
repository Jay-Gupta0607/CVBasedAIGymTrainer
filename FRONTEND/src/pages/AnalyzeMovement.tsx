import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Activity, RotateCcw, Play, Loader2, ChevronDown } from "lucide-react";
import VideoUploader from "@/components/shared/VideoUploader";
import FrameCard from "@/components/shared/FrameCard";
import { analyzeMovement, analyzeMovementWithoutTrainer, type AnalysisResponse} from "@/lib/api";
import { useToast } from "@/hooks/use-toast";
// Make sure to point this to the correct location of your Galaxy component
import { Check, ChevronsUpDown } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import Galaxy from "@/components/Galaxy";

const exercises = [
  "Barbell Biceps Curl",
  "Bench Press",
  "Chest Fly Machine",
  "Deadlift",
  "Decline Bench Press",
  "Hammer Curl",
  "Hip Thrust",
  "Incline Bench Press",
  "Lat Pulldown",
  "Lateral Raise",
  "Leg Extension",
  "Leg Raises",
  "Plank",
  "Pull Up",
  "Push-up",
  "Romanian Deadlift",
  "Russian Twist",
  "Shoulder Press",
  "Squat",
  "T Bar Row",
  "Tricep Dips",
  "Tricep Pushdown",
];

export default function AnalyzeMovement() {
  const [trainerVideo, setTrainerVideo] = useState<File | null>(null);
  const [userVideo, setUserVideo] = useState<File | null>(null);
  const [exerciseName, setExerciseName] = useState("");
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const { toast } = useToast();

  const handleAnalyze = async () => {
    if (!trainerVideo || !userVideo || !exerciseName.trim()) {
      toast({
        title: "Missing inputs",
        description: "Please provide both videos and an exercise name.",
        variant: "destructive",
      });
      return;
    }
    setLoading(true);
    setUploadProgress(0);
    try {
      const data = await analyzeMovement(trainerVideo, userVideo, exerciseName, "anksushraj2024@gmail.com", setUploadProgress);
      console.log(data);
      setResult(data);
    } catch (err: any) {
      toast({
        title: "Analysis failed",
        description: err?.message || "Could not reach the backend.",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const handleAnalyzeWithoutTrainer = async () => {
    if ( !userVideo || !exerciseName.trim()) {
      toast({
        title: "Missing inputs",
        description: "Please provide both videos and an exercise name.",
        variant: "destructive",
      });
      return;
    }
    setLoading(true);
    setUploadProgress(0);
    try {
      const data = await analyzeMovementWithoutTrainer( userVideo, exerciseName, "anksushraj2024@gmail.com", setUploadProgress);
      setResult(data);
    } catch (err: any) {
      toast({
        title: "Analysis failed",
        description: err?.message || "Could not reach the backend.",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setTrainerVideo(null);
    setUserVideo(null);
    setExerciseName("");
    setResult(null);
    setUploadProgress(0);
  };

  return (
    // 1. Outer wrapper ensuring full screen height and black background
    <div className="relative min-h-screen w-full overflow-hidden bg-black text-foreground">

      {/* 2. Background Layer: Galaxy Component */}
      <div className="absolute inset-0 z-0">
        {/* We use width/height 100% here to fill the screen instead of fixed 1080px */}
        <div style={{ width: '100%', height: '100%', position: 'relative' }}>
          <Galaxy
            starSpeed={0.5}
            density={2}
            hueShift={150}
            speed={0.9}
            glowIntensity={0.3}
            saturation={0.35}
            mouseRepulsion
            repulsionStrength={2}
            twinkleIntensity={0.25}
            rotationSpeed={0.05}
            transparent
          />
        </div>
      </div>

      {/* 3. Content Layer: Positioned relative to sit ON TOP of the background */}
      <div className="relative z-10 container mx-auto px-4 py-10">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
          <div className="flex items-center gap-3 mb-8">
            <div className="w-10 h-10 rounded-xl bg-primary/15 flex items-center justify-center neon-border backdrop-blur-sm">
              <Activity className="w-5 h-5 text-primary" />
            </div>
            <div>
              <h1 className="font-heading text-2xl font-bold text-white">Analyze Movement</h1>
              <p className="text-sm text-gray-400">Compare user form against trainer reference</p>
            </div>
          </div>

          {/* Input section */}
          {/* Added backdrop-blur to glass classes to ensure text readability over stars */}
          <div className="glass rounded-2xl p-6 mb-6 backdrop-blur-md bg-black/40 border border-white/10">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
              <VideoUploader label="Trainer Video" file={trainerVideo} onFileChange={setTrainerVideo} />
              <VideoUploader label="User Video" file={userVideo} onFileChange={setUserVideo} />
            </div>

            <div className="mb-6">
              <label className="text-sm font-medium text-gray-200 mb-2 block">
                Exercise Name
              </label>
              <Popover open={open} onOpenChange={setOpen}>
                <PopoverTrigger asChild>
                  <Button
                    variant="outline"
                    role="combobox"
                    aria-expanded={open}
                    className="w-full justify-between bg-white/5 border-white/10 text-white hover:bg-white/10 hover:text-white"
                  >
                    {exerciseName
                      ? exercises.find((exercise) => exercise === exerciseName)
                      : "Select exercise..."}
                    <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
                  </Button>
                </PopoverTrigger>
                <PopoverContent className="w-[--radix-popover-trigger-width] p-0 bg-black/90 border-white/10 text-white">
                  <Command className="bg-transparent text-white">
                    <CommandInput placeholder="Search exercise..." className="text-white placeholder:text-gray-500" />
                    <CommandList>
                      <CommandEmpty>No exercise found.</CommandEmpty>
                      <CommandGroup>
                        {exercises.map((exercise) => (
                          <CommandItem
                            key={exercise}
                            value={exercise}
                            onSelect={(currentValue) => {
                              // cmdk returns the value in lowercase, so we need to find the original casing
                              const originalVideo = exercises.find((e) => e.toLowerCase() === currentValue.toLowerCase()) || currentValue;
                              setExerciseName(originalVideo === exerciseName ? "" : originalVideo)
                              setOpen(false)
                            }}
                            className="text-gray-200 aria-selected:bg-white/10 aria-selected:text-white"
                          >
                            <Check
                              className={cn(
                                "mr-2 h-4 w-4",
                                exerciseName === exercise ? "opacity-100" : "opacity-0"
                              )}
                            />
                            {exercise}
                          </CommandItem>
                        ))}
                      </CommandGroup>
                    </CommandList>
                  </Command>
                </PopoverContent>
              </Popover>
            </div>

            <div className="flex flex-wrap gap-3">
              {/*  */}
              <button
                onClick={handleAnalyze}
                disabled={loading}
                className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-primary text-primary-foreground font-semibold hover:opacity-90 neon-glow transition-all disabled:opacity-50"
              >
                {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                Analyze Movement
              </button>
              {/*  */}
              <button
                onClick={handleAnalyzeWithoutTrainer}
                disabled={loading}
                className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-primary text-primary-foreground font-semibold hover:opacity-90 neon-glow transition-all disabled:opacity-50"
              >
                {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                Analyze Movement Without Trainer
              </button>
              {/*  */}
              <button
                onClick={handleReset}
                className="inline-flex items-center gap-2 px-6 py-3 rounded-xl glass bg-white/5 text-white font-medium hover:bg-white/10 transition-all border border-white/10"
              >
                <RotateCcw className="w-4 h-4" />
                Reset
              </button>
              <button className="inline-flex items-center gap-2 px-6 py-3 rounded-xl glass bg-white/5 text-gray-400 font-medium hover:bg-white/10 transition-all border border-white/10">
                Load Sample Videos
              </button>
            </div>
          </div>

          {/* Loading state */}
          <AnimatePresence>
            {loading && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className="glass rounded-2xl p-8 mb-6 flex flex-col items-center gap-4 backdrop-blur-md bg-black/40 border border-white/10"
              >
                <Loader2 className="w-10 h-10 text-primary animate-spin" />
                <p className="text-white font-medium">Synchronizing skeletons and analyzing posture…</p>
                <div className="w-full max-w-xs h-2 rounded-full bg-white/10 overflow-hidden">
                  <motion.div
                    className="h-full bg-primary rounded-full"
                    initial={{ width: 0 }}
                    animate={{ width: `${uploadProgress}%` }}
                    transition={{ duration: 0.3 }}
                  />
                </div>
                <p className="text-xs text-gray-400">Upload: {uploadProgress}%</p>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Results */}
          <AnimatePresence>
            {result && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                {/* Summary */}
                <div className="glass rounded-2xl p-6 mb-6 backdrop-blur-md bg-black/40 border border-white/10">
                  <h2 className="font-heading text-xl font-bold text-white mb-4 flex items-center gap-2">
                    <ChevronDown className="w-5 h-5 text-primary" />
                    Session Summary
                  </h2>
                  <p className="text-gray-200 mb-4">{result.feedback_summary}</p>
                  {result.technical_details?.length > 0 && (
                    <div>
                      <h3 className="text-sm font-semibold text-white mb-2">Technical Corrections</h3>
                      <ul className="space-y-1">
                        {result.technical_details?.map((d, i) => (
                          <li key={i} className="flex items-start gap-2 text-sm text-gray-400">
                            <span className="w-1.5 h-1.5 rounded-full bg-primary mt-1.5 shrink-0" />
                            <span>
                              <strong className="font-semibold text-gray-200">{d.title}:</strong> {d.description}
                            </span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>

                {result.reps > 0 && 
                    <div>
                      <h3>Rep count : </h3> <p> {result.reps} </p>
                    </div>
                }

                {/* Frame cards */}
                <h2 className="font-heading text-lg font-bold text-white mb-4">
                  Frame Analysis ({result.analysis?.length || 0} frames)
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {result.analysis?.map((frame, i) => (
                    <FrameCard key={frame.frame_id} frame={frame} index={i} />
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </div>
    </div>
  );
}
