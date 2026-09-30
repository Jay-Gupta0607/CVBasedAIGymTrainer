import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const ANALYZE_MOVEMENT_URL = import.meta.env.VITE_ANALYZE_MOVEMENT_URL || `${BASE_URL}/api/v1/analyze`;
const ANALYZE_MOVEMENT_URL_WITHOUT_TRAINER =
  import.meta.env.VITE_ANALYZE_MOVEMENT_URL_WITHOUT_TRAINER || `${BASE_URL}/api/v1/analyze/no-trainer`;
const GENERATE_IMAGE_URL = import.meta.env.VITE_GENERATE_IMAGE_URL || `${BASE_URL}/api/v1/generate`;
const POSE_TRANSFER_URL = import.meta.env.VITE_POSE_TRANSFER_URL || `${BASE_URL}/api/v1/generate/pose-transfer`;
const LOGIN_URL = import.meta.env.VITE_LOGIN_URL || `${BASE_URL}/api/v1/auth/login`;
const SIGNUP_URL = import.meta.env.VITE_SIGNUP_URL || `${BASE_URL}/api/v1/auth/signup`;
const CHAT_URL = import.meta.env.VITE_CHAT_URL || `${BASE_URL}/api/v1/chat`;

const api = axios.create({
  timeout: 300000,
});

// Attach the JWT to every request when the user is logged in.
const getToken = () => {
  try {
    return localStorage.getItem("access_token");
  } catch {
    return null;
  }
};

api.interceptors.request.use((config) => {
  const token = getToken();
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export interface AnalysisFrame {
  frame_id: number;
  error_score: number;
  feedback: string;
  technical_observation: string;
  user_image?: string; // legacy base64 (pre-S3 analyses)
  trainer_image?: string; // legacy base64 (pre-S3 analyses)
  user_image_url?: string; // presigned MinIO URL (current analyses)
  trainer_image_url?: string; // presigned MinIO URL (current analyses)
}

export interface AnalysisResponse {
  analysis: AnalysisFrame[];
  reps: number;
  feedback_summary: string;
  technical_details: { title: string; description: string }[];
}

export interface GenerateImageResponse {
  corrected_image: string; // base64
}

interface TaskResponse {
  task_id: string;
  status: string;
  message?: string;
}

interface StatusResponse {
  task_id: string;
  status: string;
  progress: number;
  current_stage?: string;
  error_message?: string | null;
  result?: AnalysisResponse | null;
}

/**
 * Poll the FastAPI analysis status endpoint until the job completes.
 * FastAPI's create endpoints return a task_id (202); results come from polling.
 */
async function pollAnalysis(
  taskId: string,
  onProgress?: (progress: number) => void,
  options?: { timeoutMs?: number; maxAttempts?: number }
): Promise<AnalysisResponse> {
  const timeoutMs = options?.timeoutMs ?? 5 * 60 * 1000; // 5 minutes default
  const maxAttempts = options?.maxAttempts ?? 150; // 150 * 2s = 5 min
  const startTime = Date.now();
  let attempt = 0;

  for (;;) {
    attempt++;
    const response = await api.get<StatusResponse>(`${BASE_URL}/api/v1/analyze/${taskId}/status`);
    const data = response.data;

    if (onProgress) {
      onProgress(data.progress ?? 0);
    }

    if (data.status === "completed" && data.result) {
      return data.result;
    }
    if (data.status === "failed") {
      throw new Error(data.error_message || "Analysis failed");
    }

    // Timeout / max attempts guard
    if (Date.now() - startTime >= timeoutMs) {
      throw new Error("Analysis timed out. The job may still be running — check history later.");
    }
    if (attempt >= maxAttempts) {
      throw new Error("Max polling attempts reached. The job may still be running — check history later.");
    }

    await new Promise((resolve) => setTimeout(resolve, 2000));
  }
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

  const response = await api.post<TaskResponse>(ANALYZE_MOVEMENT_URL, formData, {
    headers: { "Content-Type": "multipart/form-data" },
    onUploadProgress: (e) => {
      if (e.total && onProgress) {
        onProgress(Math.round((e.loaded / e.total) * 100));
      }
    },
  });

  return pollAnalysis(response.data.task_id, onProgress);
}

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

  const response = await api.post<TaskResponse>(ANALYZE_MOVEMENT_URL_WITHOUT_TRAINER, formData, {
    headers: { "Content-Type": "multipart/form-data" },
    onUploadProgress: (e) => {
      if (e.total && onProgress) {
        onProgress(Math.round((e.loaded / e.total) * 100));
      }
    },
  });

  return pollAnalysis(response.data.task_id, onProgress);
}

export async function generateImage(image: File, prompt: string): Promise<Blob> {
  const formData = new FormData();
  formData.append("image", image);
  formData.append("prompt", prompt);

  if (!GENERATE_IMAGE_URL) {
    throw new Error("Generate Image URL is not configured");
  }

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

export async function signupUser(email: string, password: string, fullName?: string) {
  const response = await api.post(SIGNUP_URL, { email, password, full_name: fullName });
  return response.data;
}

export async function sendChatMessage(message: string) {
  const response = await api.post(CHAT_URL, { message });
  return response.data;
}

/**
 * Turn an axios/fetch error into a human-readable message.
 * FastAPI returns {"detail": "..."} on errors; a 503 from the image-generation
 * endpoints specifically means the external service isn't configured.
 */
export function getErrorMessage(err: unknown): string {
  const response = (err as { response?: { status?: number; data?: { detail?: string } } })?.response;

  if (response?.status === 503) {
    const detail = response.data?.detail ?? "";
    if (/not configured/i.test(detail)) {
      return (
        "AI image generation isn't configured on this server yet. " +
        "This feature needs a Modal deployment to be wired up — movement analysis still works."
      );
    }
    return detail || "Service temporarily unavailable.";
  }

  const detail = response?.data?.detail;
  if (detail) return detail;
  return (err as Error)?.message || "Something went wrong.";
}

export async function generatePoseTransfer(
  userImage: File,
  trainerImage: File,
  exerciseName: string,
  frameId: number
): Promise<Blob> {
  const formData = new FormData();
  formData.append("user_image", userImage);
  formData.append("trainer_image", trainerImage);
  formData.append("exercise_name", exerciseName);
  formData.append("frame_id", frameId.toString());

  if (!POSE_TRANSFER_URL) {
    throw new Error("Pose Transfer URL is not configured");
  }

  const response = await api.post(POSE_TRANSFER_URL, formData, {
    headers: { "Content-Type": "multipart/form-data" },
    responseType: "blob",
    timeout: 180000,
  });

  return response.data;
}

export { CHAT_URL, LOGIN_URL, SIGNUP_URL, POSE_TRANSFER_URL };
export default api;