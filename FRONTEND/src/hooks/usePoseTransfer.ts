import { useCallback, useState, useEffect } from "react";
import { generatePoseTransfer, getErrorMessage } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";

export interface PoseTransferResult {
  imageUrl: string; // Object URL for the generated image
  frameId: number;
  exerciseName: string;
}

export function usePoseTransfer() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PoseTransferResult | null>(null);
  const { toast } = useToast();

  // Cleanup Object URL on unmount
  useEffect(() => {
    return () => {
      if (result?.imageUrl) {
        URL.revokeObjectURL(result.imageUrl);
      }
    };
  }, []); // Only run on mount/unmount

  const generate = useCallback(
    async (
      userImage: File,
      trainerImage: File,
      exerciseName: string,
      frameId: number
    ): Promise<PoseTransferResult | null> => {
      setLoading(true);
      setResult(null);

      try {
        const blob = await generatePoseTransfer(userImage, trainerImage, exerciseName, frameId);
        const imageUrl = URL.createObjectURL(blob);

        const transferResult: PoseTransferResult = {
          imageUrl,
          frameId,
          exerciseName,
        };

        setResult(transferResult);
        toast({
          title: "Pose transfer generated",
          description: "Your corrected form visualization is ready.",
        });

        return transferResult;
      } catch (err) {
        toast({
          title: "Generation unavailable",
          description: getErrorMessage(err),
          variant: "destructive",
        });
        return null;
      } finally {
        setLoading(false);
      }
    },
    [toast]
  );

  const clear = useCallback(() => {
    setResult((prev) => {
      if (prev?.imageUrl) {
        URL.revokeObjectURL(prev.imageUrl);
      }
      return null;
    });
  }, []);

  return {
    loading,
    result,
    generate,
    clear,
  };
}