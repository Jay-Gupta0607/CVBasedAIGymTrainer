import { motion, AnimatePresence } from "framer-motion";
import { Loader2, Image as ImageIcon, ArrowRightLeft, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { usePoseTransfer, type PoseTransferResult } from "@/hooks/usePoseTransfer";
import { useState, useCallback, useMemo } from "react";

interface PoseTransferButtonProps {
  userImage: string; // base64 from frame OR object URL
  trainerImage?: string; // base64 trainer reference (from analysis or stored) OR object URL
  exerciseName: string;
  frameId: number;
  onGenerated?: (result: PoseTransferResult) => void;
}

async function imageSourceToFile(src: string, filename: string): Promise<File> {
  // If it's already a data URL, convert directly
  if (src.startsWith("data:")) {
    const arr = src.split(",");
    const mime = arr[0].match(/:(.*?);/)![1];
    const bstr = atob(arr[1]);
    const u8arr = new Uint8Array(bstr.length);
    for (let i = 0; i < bstr.length; i++) {
      u8arr[i] = bstr.charCodeAt(i);
    }
    return new File([u8arr], filename, { type: mime });
  }
  // If it's an object URL or HTTP URL, fetch it
  const response = await fetch(src);
  const blob = await response.blob();
  return new File([blob], filename, { type: blob.type || "image/jpeg" });
}

export default function PoseTransferButton({
  userImage,
  trainerImage,
  exerciseName,
  frameId,
  onGenerated,
}: PoseTransferButtonProps) {
  const { loading, result, generate, clear } = usePoseTransfer();
  const [showResult, setShowResult] = useState(false);

  // Memoize file conversion to avoid re-fetching on every render
  const userFilePromise = useMemo(() => imageSourceToFile(userImage, `user_frame_${frameId}.jpg`), [userImage, frameId]);
  const trainerFilePromise = useMemo(() =>
    imageSourceToFile(trainerImage || userImage, `trainer_ref_${frameId}.jpg`),
    [trainerImage, userImage, frameId]
  );

  const handleGenerate = useCallback(async () => {
    try {
      const [userFile, trainerFile] = await Promise.all([userFilePromise, trainerFilePromise]);
      const transferResult = await generate(userFile, trainerFile, exerciseName, frameId);
      if (transferResult && onGenerated) {
        onGenerated(transferResult);
      }
      setShowResult(true);
    } catch (err) {
      console.error("Failed to prepare images for pose transfer:", err);
    }
  }, [userFilePromise, trainerFilePromise, generate, exerciseName, frameId, onGenerated]);

  if (!trainerImage && !result) {
    // Show disabled state if no trainer reference available
    return (
      <Button
        variant="outline"
        disabled
        className="w-full gap-2 text-xs px-3 py-2"
        title="Trainer reference image required for pose transfer"
      >
        <ArrowRightLeft className="w-3.5 h-3.5 opacity-50" />
        <span>No Trainer Ref</span>
      </Button>
    );
  }

  return (
    <div className="w-full">
      {!result ? (
        <Button
          onClick={handleGenerate}
          disabled={loading}
          className="w-full gap-2 text-xs px-3 py-2"
        >
          {loading ? (
            <>
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
              <span>Generating...</span>
            </>
          ) : (
            <>
              <Sparkles className="w-3.5 h-3.5" />
              <span>Generate Pose Transfer</span>
            </>
          )}
        </Button>
      ) : (
        <div className="space-y-2">
          {/* Action buttons */}
          <div className="flex gap-2">
            <Button
              variant={showResult ? "default" : "outline"}
              onClick={() => setShowResult(true)}
              className="flex-1 gap-1.5 text-xs px-3 py-2"
            >
              <ImageIcon className="w-3.5 h-3.5" />
              <span>View Result</span>
            </Button>
            <Button
              variant="outline"
              onClick={() => {
                clear();
                setShowResult(false);
              }}
              className="gap-1.5 text-xs px-3 py-2"
            >
              <ArrowRightLeft className="w-3.5 h-3.5" />
              <span>Regenerate</span>
            </Button>
          </div>

          {/* Generated image display */}
          <AnimatePresence>
            {showResult && result && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className="rounded-lg overflow-hidden border border-border/50"
              >
                <div className="text-xs uppercase tracking-widest text-muted-foreground px-2 py-1 bg-secondary/50 flex items-center gap-1">
                  <Sparkles className="w-3 h-3 text-primary" />
                  Pose Transfer — Your Body + Trainer's Form
                </div>
                <img
                  src={result.imageUrl}
                  alt="Pose transfer result"
                  className="w-full aspect-video object-cover"
                />
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}
    </div>
  );
}