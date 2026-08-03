import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Image, Loader2, Download, RefreshCw } from "lucide-react";
import ImageUploader from "@/components/shared/ImageUploader";
import { generateImage } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";

export default function GenerateImage() {
  const [file, setFile] = useState<File | null>(null);

  const [loading, setLoading] = useState(false);
  const [correctedImage, setCorrectedImage] = useState<string | null>(null);

  const { toast } = useToast();


  const originalPreview = file ? URL.createObjectURL(file) : null;

  const handleGenerate = async () => {
    if (!file) {
      toast({
        title: "Missing input",
        description: "Please provide an image.",
        variant: "destructive",
      });
      return;
    }
    setLoading(true);
    try {
      // Use a default prompt as requested/implied by removal of inputs. 
      // User curl example used "rod", but "General" or similar is safer if "rod" isn't magic.
      // I'll use "fix form" as a generic prompt to satisfy the requirement.
      const blob = await generateImage(file, "rod");
      const url = URL.createObjectURL(blob);
      setCorrectedImage(url);
      // setShowCompare(false); // Removed as state is deleted
    } catch (err: any) {
      toast({
        title: "Generation failed",
        description: err?.message || "Could not reach the backend.",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = () => {
    if (!correctedImage) return;
    const a = document.createElement("a");
    a.href = correctedImage;
    a.download = `corrected_image.jpg`;
    a.click();
  };



  return (
    <div className="container mx-auto px-4 py-10">
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <div className="flex items-center gap-3 mb-8">
          <div className="w-10 h-10 rounded-xl bg-primary/15 flex items-center justify-center neon-border">
            <Image className="w-5 h-5 text-primary" />
          </div>
          <div>
            <h1 className="font-heading text-2xl font-bold text-foreground">Generate Correct Form</h1>
            <p className="text-sm text-muted-foreground">AI-powered form correction image</p>
          </div>
        </div>

        {/* Inputs */}
        <div className="glass rounded-2xl p-6 mb-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            <ImageUploader label="User Image / Frame" file={file} onFileChange={setFile} />
          </div>

          <div className="flex flex-wrap gap-3">
            <button
              onClick={handleGenerate}
              disabled={loading}
              className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-primary text-primary-foreground font-semibold hover:opacity-90 neon-glow transition-all disabled:opacity-50"
            >
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Image className="w-4 h-4" />}
              Generate Correct Form Image
            </button>
            {correctedImage && (
              <>

                <button
                  onClick={handleDownload}
                  className="inline-flex items-center gap-2 px-6 py-3 rounded-xl glass text-foreground font-medium hover:bg-secondary transition-all"
                >
                  <Download className="w-4 h-4" />
                  Download
                </button>
                <button
                  onClick={handleGenerate}
                  disabled={loading}
                  className="inline-flex items-center gap-2 px-6 py-3 rounded-xl glass text-muted-foreground font-medium hover:bg-secondary transition-all disabled:opacity-50"
                >
                  <RefreshCw className="w-4 h-4" />
                  Regenerate
                </button>
              </>
            )}
          </div>
        </div>

        {/* Loading */}
        <AnimatePresence>
          {loading && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="glass rounded-2xl p-8 mb-6 flex flex-col items-center gap-4"
            >
              <Loader2 className="w-10 h-10 text-primary animate-spin" />
              <p className="text-foreground font-medium">Generating corrected form image…</p>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Output */}
        <AnimatePresence>
          {correctedImage && !loading && (
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
              <div className="glass rounded-2xl overflow-hidden neon-border p-4">
                <h2 className="font-heading text-lg font-bold text-foreground mb-4">Generated Result</h2>
                <img src={correctedImage} alt="Corrected" className="w-full h-auto rounded-xl object-contain bg-black/30" />
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </div>
  );
}
