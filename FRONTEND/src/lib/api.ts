import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_BASE_URL || "https://independently-unapplauded-azzie.ngrok-free.dev";

const ANALYZE_MOVEMENT_URL = import.meta.env.VITE_ANALYZE_MOVEMENT_URL || `${BASE_URL}/analyze_movement`;
const ANALYZE_MOVEMENT_URL_WITHOUT_TRAINER = import.meta.env.VITE_ANALYZE_MOVEMENT_URL_WITHOUT_TRAINER || `${BASE_URL}/analyze_without_video`;
const GENERATE_IMAGE_URL = import.meta.env.VITE_GENERATE_IMAGE_URL || "https://ankushraj2024--maskedbar-pipeline-full-imagepipeline-web-4b6c7d.modal.run";
const LOGIN_URL = import.meta.env.VITE_LOGIN_URL || `${BASE_URL}/login`;
const SIGNUP_URL = import.meta.env.VITE_SIGNUP_URL || `${BASE_URL}/signup`;
const CHAT_URL = import.meta.env.VITE_CHAT_URL || `${BASE_URL}/chat`;

const api = axios.create({
  timeout: 300000,
});

export interface AnalysisFrame {
  frame_id: number;
  error_score: number;
  feedback: string;
  technical_observation: string;
  user_image: string; // base64
  trainer_image: string; // base64
}

export interface AnalysisResponse {
  analysis: AnalysisFrame[];

  
  reps: number,
  feedback_summary: string;
  technical_details: { title: string; description: string }[];
}

export interface GenerateImageResponse {
  corrected_image: string; // base64
}

export async function analyzeMovement(
  trainerVideo: File,
  userVideo: File,
  exerciseName: string,
  email: string,
  onProgress?: (progress: number) => void
): Promise<AnalysisResponse> {
  const formData = new FormData();
  formData.append("trainer_video", trainerVideo);
  formData.append("user_video", userVideo);
  formData.append("exercise_name", exerciseName);
  formData.append("email", email);

  // Post directly to the function URL, no path appending
  const response = await api.post<AnalysisResponse>(ANALYZE_MOVEMENT_URL, formData, {
    headers: { "Content-Type": "multipart/form-data" },
    onUploadProgress: (e) => {
      if (e.total && onProgress) {
        onProgress(Math.round((e.loaded / e.total) * 100));
      }
    },
  });

  return response.data;
}

// ---------- // 
export async function analyzeMovementWithoutTrainer(
  userVideo: File,
  exerciseName: string,
  email: string,
  onProgress?: (progress: number) => void
): Promise<AnalysisResponse> {
  const formData = new FormData();
  formData.append("user_video", userVideo);
  formData.append("exercise_name", exerciseName);
  formData.append("email", email);

  // Post directly to the function URL, no path appending
  const response = await api.post<AnalysisResponse>(ANALYZE_MOVEMENT_URL_WITHOUT_TRAINER, formData, {
    headers: { "Content-Type": "multipart/form-data" },
    onUploadProgress: (e) => {
      if (e.total && onProgress) {
        onProgress(Math.round((e.loaded / e.total) * 100));
      }
    },
  });

  return response.data;
}
// ----------- //

export async function generateImage(
  image: File,
  prompt: string
): Promise<Blob> {
  const formData = new FormData();
  formData.append("image", image);
  formData.append("prompt", prompt);

  if (!GENERATE_IMAGE_URL) {
    throw new Error("Generate Image URL is not configured");
  }

  // Post directly to the function URL
  const response = await api.post(GENERATE_IMAGE_URL, formData, {
    headers: { "Content-Type": "multipart/form-data" },
    responseType: "blob",
  });

  return response.data;
}

export async function loginUser(email: string, password: string) {
  const response = await api.post(LOGIN_URL, { email, password });
  return response.data;
}

export async function signupUser(email: string, password: string) {
  const response = await api.post(SIGNUP_URL, { email, password });
  return response.data;
}

export async function sendChatMessage(message: string) {
  const response = await api.post(CHAT_URL, { message });
  return response.data;
}

export { CHAT_URL, LOGIN_URL, SIGNUP_URL };
export default api;
