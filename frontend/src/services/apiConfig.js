const configuredApiUrl = import.meta.env.VITE_API_URL?.trim();

// Keep local development convenient, but never bake a localhost API address
// into a production build when the separately deployed backend is not ready.
export const API_BASE_URL = (
  configuredApiUrl || (import.meta.env.DEV ? "http://localhost:8000" : "")
).replace(/\/+$/, "");

export function requireApiBaseUrl() {
  if (!API_BASE_URL) {
    throw new Error(
      "The backend API is not configured. Set VITE_API_URL after deploying the backend."
    );
  }

  return API_BASE_URL;
}
