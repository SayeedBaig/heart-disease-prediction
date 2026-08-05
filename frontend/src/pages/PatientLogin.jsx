import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import {
  Mail,
  Lock,
  Eye,
  EyeOff,
  HeartPulse,
  ArrowRight,
  ArrowLeft
} from "lucide-react";
import { AppLayout } from "../components/ui/AppLayout";
import { Container } from "../components/ui/Container";
import { Section } from "../components/ui/Section";
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from "../components/ui/Card";
import { Button } from "../components/ui/Button";
import { Input } from "../components/ui/Input";
import api from "../services/api";
import { clearPatientSession } from "../services/portalService";

function PatientLogin() {
  const navigate = useNavigate();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleLogin = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    clearPatientSession();

    try {
      const loginResponse = await api.post("/patients/login", {
        email,
        password,
      });

      const token = loginResponse.data.access_token;
      localStorage.setItem("access_token", token);

      const profileResponse = await api.get("/patients/me", {
        headers: { Authorization: `Bearer ${token}` },
      });
      const patientData = profileResponse.data;

      localStorage.setItem("patient", JSON.stringify(patientData));
      localStorage.setItem("cardio-patient", JSON.stringify(patientData));

      navigate("/patient/dashboard");
    } catch (err) {
      console.error("Patient Login error:", err);
      setError(err.response?.data?.detail || "Unable to sign in. Please check your connection and try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <AppLayout>
      <Section className="min-h-[85vh] flex items-center justify-center bg-secondary/10 py-12">
        <Container className="flex justify-center">
          <Card className="w-full max-w-[440px] shadow-lg border-border bg-background">
            <CardHeader className="text-center pb-6">
              <div className="mx-auto h-12 w-12 rounded-xl bg-primary/10 flex items-center justify-center mb-4 border border-primary/20">
                <HeartPulse className="h-6 w-6 text-primary" />
              </div>
              <CardTitle className="text-2xl font-bold">Patient Login</CardTitle>
              <CardDescription>
                Log in to access your cardiovascular health suite
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleLogin} className="space-y-5">
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">Email Address</label>
                  <div className="relative">
                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-muted-foreground">
                      <Mail className="h-4 w-4" />
                    </div>
                    <Input
                      type="email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      required
                      placeholder="patient@example.com"
                      className="pl-9 h-10"
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <label className="text-sm font-medium text-foreground">Password</label>
                    <a href="#" className="text-xs font-medium text-primary hover:underline">
                      Forgot Password?
                    </a>
                  </div>
                  <div className="relative">
                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-muted-foreground">
                      <Lock className="h-4 w-4" />
                    </div>
                    <Input
                      type={showPassword ? "text" : "password"}
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      required
                      placeholder="Enter your password"
                      className="pl-9 pr-10 h-10"
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassword(!showPassword)}
                      className="absolute inset-y-0 right-0 pr-3 flex items-center text-muted-foreground hover:text-foreground transition-colors"
                    >
                      {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </button>
                  </div>
                </div>

                <div className="min-h-[24px]">
                  {error && (
                    <div className="p-2.5 rounded-lg bg-destructive/10 border border-destructive/20 text-destructive text-xs font-medium text-center">
                      {error}
                    </div>
                  )}
                </div>

                <Button
                  type="submit"
                  disabled={loading}
                  className="w-full h-11 font-semibold text-sm"
                >
                  {loading ? "Logging in..." : "Log In"}
                  {!loading && <ArrowRight className="ml-2 h-4 w-4" />}
                </Button>
              </form>

              <div className="relative my-6">
                <div className="absolute inset-0 flex items-center">
                  <div className="w-full border-t border-border" />
                </div>
                <div className="relative flex justify-center text-xs uppercase">
                  <span className="bg-background px-2 text-muted-foreground font-medium">Or</span>
                </div>
              </div>

              <Button
                variant="outline"
                className="w-full h-11 font-semibold text-sm"
                onClick={() => navigate("/patient/signup")}
              >
                Create a new account
              </Button>
            </CardContent>
            <CardFooter className="flex justify-center border-t border-border pt-6 mt-2 pb-6">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => navigate("/get-started")}
                className="text-muted-foreground"
              >
                <ArrowLeft className="mr-2 h-4 w-4" />
                Back to Portal Selection
              </Button>
            </CardFooter>
          </Card>
        </Container>
      </Section>
    </AppLayout>
  );
}

export default PatientLogin;
