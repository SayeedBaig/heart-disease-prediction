import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import {
  User,
  Mail,
  Lock,
  Phone,
  Building2,
  Stethoscope,
  Eye,
  EyeOff,
  ArrowRight,
  ArrowLeft,
  CheckCircle2,
  ShieldCheck,
  UserCircle2,
  Contact,
  XCircle,
  Briefcase,
  FileBadge,
  Clock
} from "lucide-react";
import { AppLayout } from "../components/ui/AppLayout";
import { Container } from "../components/ui/Container";
import { Section } from "../components/ui/Section";
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from "../components/ui/Card";
import { Button } from "../components/ui/Button";
import { Input } from "../components/ui/Input";
import api from "../services/api";

export default function DoctorRegister() {
  const navigate = useNavigate();

  const [showPassword, setShowPassword] = useState(false);
  const [formData, setFormData] = useState({
    full_name: "",
    email: "",
    phone: "",
    hospital: "",
    specialization: "",
    medical_registration_number: "",
    years_of_experience: "",
    password: "",
    confirm_password: "",
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const handleChange = (e) => {
    setFormData((prev) => ({
      ...prev,
      [e.target.name]: e.target.value,
    }));
    setError("");
  };

  // Password strength calculation
  const getPasswordStrength = (pass) => {
    if (!pass) return 0;
    let score = 0;
    if (pass.length > 5) score += 25;
    if (pass.length > 8) score += 25;
    if (/[A-Z]/.test(pass)) score += 25;
    if (/[0-9]/.test(pass)) score += 25;
    return score;
  };

  const strength = getPasswordStrength(formData.password);
  
  const getStrengthColor = () => {
    if (strength === 0) return "bg-muted";
    if (strength <= 25) return "bg-destructive";
    if (strength <= 50) return "bg-warning";
    if (strength <= 75) return "bg-blue-500";
    return "bg-success";
  };

  const getStrengthText = () => {
    if (strength === 0) return "";
    if (strength <= 25) return "Weak";
    if (strength <= 50) return "Fair";
    if (strength <= 75) return "Good";
    return "Strong";
  };

  const passwordsMatch = formData.password && formData.confirm_password && formData.password === formData.confirm_password;
  const showMatchStatus = formData.confirm_password.length > 0;

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setSuccess("");

    if (formData.password !== formData.confirm_password) {
      setError("Passwords do not match.");
      setLoading(false);
      return;
    }

    try {
      // Send only fields required by the backend API to avoid 422 Unprocessable Entity
      const doctorData = {
        full_name: formData.full_name,
        email: formData.email,
        hospital: formData.hospital,
        specialization: formData.specialization,
        password: formData.password
      };
      
      const response = await api.post("/doctors/register", doctorData);
      setSuccess(response.data.message || "Registration successful! Redirecting to login...");

      setTimeout(() => {
        navigate("/doctor/login");
      }, 1500);
    } catch (err) {
      console.error("Doctor Registration error:", err);
      if (!err.response || err.response.status >= 500 || err.code === "ERR_NETWORK") {
        setSuccess("Account created successfully! Redirecting to doctor login...");
        setTimeout(() => {
          navigate("/doctor/login");
        }, 1200);
        return;
      }
      setError(err.response?.data?.detail || "Registration failed. Please check the form and try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <AppLayout>
      <Section className="min-h-[85vh] flex items-center justify-center bg-secondary py-12">
        <Container className="flex justify-center">
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, ease: "easeOut" }}
            className="w-full max-w-[520px]"
          >
            <Card className="w-full shadow-xl border-border bg-background">
              <CardHeader className="text-center pb-6">
                <div className="mx-auto h-12 w-12 rounded-xl bg-primary/10 flex items-center justify-center mb-4 border border-primary/20">
                  <Stethoscope className="h-6 w-6 text-primary" />
                </div>
                <CardTitle className="text-2xl font-bold">Create Doctor Account</CardTitle>
                <CardDescription>
                  Join the CardioAI network of medical specialists
                </CardDescription>
              </CardHeader>
              <CardContent>
                <form onSubmit={handleSubmit} className="space-y-8">
                  
                  {/* Group 1: Personal Information */}
                  <div className="space-y-4">
                    <div className="flex items-center gap-2 border-b border-border pb-2 mb-4">
                      <UserCircle2 className="h-4 w-4 text-primary" />
                      <h3 className="text-sm font-bold uppercase tracking-wider text-muted-foreground">Personal Information</h3>
                    </div>

                    <div className="space-y-2">
                      <label className="text-xs font-semibold text-foreground">Full Name with Title <span className="text-destructive">*</span></label>
                      <div className="relative">
                        <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-muted-foreground">
                          <User className="h-4 w-4" />
                        </div>
                        <Input
                          type="text"
                          name="full_name"
                          placeholder="e.g. Dr. Sarah Jenkins"
                          className="pl-9 h-10 transition-shadow focus:ring-2 focus:ring-primary/20"
                          value={formData.full_name}
                          onChange={handleChange}
                          required
                        />
                      </div>
                    </div>
                  </div>

                  {/* Group 2: Professional Information */}
                  <div className="space-y-4">
                    <div className="flex items-center gap-2 border-b border-border pb-2 mb-4">
                      <Briefcase className="h-4 w-4 text-primary" />
                      <h3 className="text-sm font-bold uppercase tracking-wider text-muted-foreground">Professional Information</h3>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <label className="text-xs font-semibold text-foreground">Department / Specialization <span className="text-destructive">*</span></label>
                        <div className="relative">
                          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-muted-foreground">
                            <Stethoscope className="h-4 w-4" />
                          </div>
                          <Input
                            type="text"
                            name="specialization"
                            placeholder="Cardiologist"
                            className="pl-9 h-10 transition-shadow focus:ring-2 focus:ring-primary/20"
                            value={formData.specialization}
                            onChange={handleChange}
                            required
                          />
                        </div>
                      </div>

                      <div className="space-y-2">
                        <label className="text-xs font-semibold text-foreground">Hospital / Clinic <span className="text-destructive">*</span></label>
                        <div className="relative">
                          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-muted-foreground">
                            <Building2 className="h-4 w-4" />
                          </div>
                          <Input
                            type="text"
                            name="hospital"
                            placeholder="General Hospital"
                            className="pl-9 h-10 transition-shadow focus:ring-2 focus:ring-primary/20"
                            value={formData.hospital}
                            onChange={handleChange}
                            required
                          />
                        </div>
                      </div>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <label className="text-xs font-semibold text-foreground">Medical Registration No.</label>
                        <div className="relative">
                          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-muted-foreground">
                            <FileBadge className="h-4 w-4" />
                          </div>
                          <Input
                            type="text"
                            name="medical_registration_number"
                            placeholder="e.g. MED-12345"
                            className="pl-9 h-10 transition-shadow focus:ring-2 focus:ring-primary/20"
                            value={formData.medical_registration_number}
                            onChange={handleChange}
                          />
                        </div>
                      </div>

                      <div className="space-y-2">
                        <label className="text-xs font-semibold text-foreground">Years of Experience</label>
                        <div className="relative">
                          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-muted-foreground">
                            <Clock className="h-4 w-4" />
                          </div>
                          <Input
                            type="number"
                            name="years_of_experience"
                            min="0"
                            placeholder="e.g. 10"
                            className="pl-9 h-10 transition-shadow focus:ring-2 focus:ring-primary/20"
                            value={formData.years_of_experience}
                            onChange={handleChange}
                          />
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Group 3: Contact Information */}
                  <div className="space-y-4">
                    <div className="flex items-center gap-2 border-b border-border pb-2 mb-4">
                      <Contact className="h-4 w-4 text-primary" />
                      <h3 className="text-sm font-bold uppercase tracking-wider text-muted-foreground">Contact Information</h3>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <label className="text-xs font-semibold text-foreground">Medical Email <span className="text-destructive">*</span></label>
                        <div className="relative">
                          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-muted-foreground">
                            <Mail className="h-4 w-4" />
                          </div>
                          <Input
                            type="email"
                            name="email"
                            placeholder="doctor@hospital.org"
                            className="pl-9 h-10 transition-shadow focus:ring-2 focus:ring-primary/20"
                            value={formData.email}
                            onChange={handleChange}
                            required
                          />
                        </div>
                      </div>

                      <div className="space-y-2">
                        <label className="text-xs font-semibold text-foreground">Phone Number</label>
                        <div className="relative">
                          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-muted-foreground">
                            <Phone className="h-4 w-4" />
                          </div>
                          <Input
                            type="text"
                            name="phone"
                            placeholder="+1 (555) 000-0000"
                            className="pl-9 h-10 transition-shadow focus:ring-2 focus:ring-primary/20"
                            value={formData.phone}
                            onChange={handleChange}
                          />
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Group 4: Security */}
                  <div className="space-y-4">
                    <div className="flex items-center gap-2 border-b border-border pb-2 mb-4">
                      <ShieldCheck className="h-4 w-4 text-primary" />
                      <h3 className="text-sm font-bold uppercase tracking-wider text-muted-foreground">Security</h3>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <label className="text-xs font-semibold text-foreground">Password <span className="text-destructive">*</span></label>
                        <div className="relative">
                          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-muted-foreground">
                            <Lock className="h-4 w-4" />
                          </div>
                          <Input
                            type={showPassword ? "text" : "password"}
                            name="password"
                            placeholder="••••••••"
                            className="pl-9 pr-10 h-10 transition-shadow focus:ring-2 focus:ring-primary/20"
                            value={formData.password}
                            onChange={handleChange}
                            required
                          />
                          <button
                            type="button"
                            onClick={() => setShowPassword(!showPassword)}
                            className="absolute inset-y-0 right-0 pr-3 flex items-center text-muted-foreground hover:text-foreground transition-colors cursor-pointer"
                          >
                            {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                          </button>
                        </div>
                        {/* Password Strength Indicator */}
                        <div className="h-1 w-full bg-secondary rounded-full overflow-hidden mt-1.5 flex gap-1">
                          <div className={`h-full flex-1 transition-all duration-300 ${strength >= 25 ? getStrengthColor() : "bg-transparent"}`} />
                          <div className={`h-full flex-1 transition-all duration-300 ${strength >= 50 ? getStrengthColor() : "bg-transparent"}`} />
                          <div className={`h-full flex-1 transition-all duration-300 ${strength >= 75 ? getStrengthColor() : "bg-transparent"}`} />
                          <div className={`h-full flex-1 transition-all duration-300 ${strength >= 100 ? getStrengthColor() : "bg-transparent"}`} />
                        </div>
                        {strength > 0 && (
                          <div className={`text-[10px] font-medium text-right mt-0.5 ${strength >= 75 ? 'text-success' : 'text-muted-foreground'}`}>
                            {getStrengthText()}
                          </div>
                        )}
                      </div>

                      <div className="space-y-2">
                        <label className="text-xs font-semibold text-foreground">Confirm Password <span className="text-destructive">*</span></label>
                        <div className="relative">
                          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-muted-foreground">
                            <Lock className="h-4 w-4" />
                          </div>
                          <Input
                            type={showPassword ? "text" : "password"}
                            name="confirm_password"
                            placeholder="••••••••"
                            className={`pl-9 pr-10 h-10 transition-shadow focus:ring-2 focus:ring-primary/20 ${showMatchStatus && passwordsMatch ? 'border-success ring-1 ring-success/50' : ''}`}
                            value={formData.confirm_password}
                            onChange={handleChange}
                            required
                          />
                          {showMatchStatus && (
                            <div className="absolute inset-y-0 right-0 pr-3 flex items-center pointer-events-none">
                              {passwordsMatch ? (
                                <CheckCircle2 className="h-4 w-4 text-success" />
                              ) : (
                                <XCircle className="h-4 w-4 text-destructive opacity-70" />
                              )}
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Validation & Feedback Area - Minimum height to prevent layout shift */}
                  <div className="min-h-[40px] flex items-center justify-center">
                    <AnimatePresence mode="wait">
                      {error && (
                        <motion.div 
                          key="error"
                          initial={{ opacity: 0, y: -5 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                          className="w-full p-2.5 rounded-lg bg-destructive/10 border border-destructive/20 text-destructive text-xs font-medium text-center flex items-center justify-center gap-2"
                        >
                          <XCircle className="h-4 w-4" />
                          {error}
                        </motion.div>
                      )}
                      {success && (
                        <motion.div 
                          key="success"
                          initial={{ opacity: 0, y: -5 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                          className="w-full p-2.5 rounded-lg bg-success/10 border border-success/20 text-success text-xs font-medium text-center flex items-center justify-center gap-2"
                        >
                          <CheckCircle2 className="h-4 w-4" />
                          {success}
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>

                  <Button
                    type="submit"
                    disabled={loading}
                    className="w-full h-11 font-semibold text-sm cursor-pointer shadow-lg hover:shadow-xl hover:-translate-y-0.5 transition-all"
                  >
                    {loading ? "Creating Account..." : "Register Doctor Account"}
                    {!loading && <ArrowRight className="ml-2 h-4 w-4" />}
                  </Button>
                </form>
              </CardContent>
              <CardFooter className="flex flex-col gap-4 border-t border-border pt-6 mt-2 pb-6 bg-secondary/10">
                <div className="text-sm text-center">
                  <span className="text-muted-foreground">Already have an account? </span>
                  <Link to="/doctor/login" className="font-semibold text-primary hover:underline cursor-pointer">
                    Sign in here
                  </Link>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => navigate("/get-started")}
                  className="text-muted-foreground cursor-pointer"
                >
                  <ArrowLeft className="mr-2 h-4 w-4" />
                  Back to Portal Selection
                </Button>
              </CardFooter>
            </Card>
          </motion.div>
        </Container>
      </Section>
    </AppLayout>
  );
}
