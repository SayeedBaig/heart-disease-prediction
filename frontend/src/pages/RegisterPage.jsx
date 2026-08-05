import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import {
  HeartPulse,
  User,
  Mail,
  Phone,
  Calendar,
  UserCircle2,
  Contact,
  ArrowRight,
  XCircle
} from "lucide-react";
import { AppLayout } from "../components/ui/AppLayout";
import { Container } from "../components/ui/Container";
import { Section } from "../components/ui/Section";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "../components/ui/Card";
import { Button } from "../components/ui/Button";
import { Input } from "../components/ui/Input";
import { Select } from "../components/ui/Select";
import api from "../services/api";

function RegisterPage() {
  const navigate = useNavigate();

  const [formData, setFormData] = useState({
    full_name: "",
    email: "",
    phone: "",
    gender: "",
    date_of_birth: "",
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleRegister = async (e) => {
    if (e && e.preventDefault) {
      e.preventDefault();
    }
    setLoading(true);
    setError("");

    try {
      const response = await api.post("/patients/register", formData);

      // Save patient information
      localStorage.setItem("patient_id", response.data.patient_id);
      localStorage.setItem(
        "patient_name",
        response.data.full_name
      );
      localStorage.setItem("patient_email", formData.email);

      navigate("/dashboard");

    } catch (err) {
      console.error(err);

      setError(
        err.response?.data?.detail ||
        "Registration failed."
      );

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
                  <HeartPulse className="h-6 w-6 text-primary" />
                </div>
                <CardTitle className="text-2xl font-bold">Patient Registration</CardTitle>
                <CardDescription>
                  Register details before starting your AI diagnosis
                </CardDescription>
              </CardHeader>
              <CardContent>
                <form onSubmit={handleRegister} className="space-y-8">
                  {/* Group 1: Personal Information */}
                  <div className="space-y-4">
                    <div className="flex items-center gap-2 border-b border-border pb-2 mb-4">
                      <UserCircle2 className="h-4 w-4 text-primary" />
                      <h3 className="text-sm font-bold uppercase tracking-wider text-muted-foreground">Personal Information</h3>
                    </div>

                    <div className="space-y-2">
                      <label className="text-xs font-semibold text-foreground">Full Name</label>
                      <div className="relative">
                        <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-muted-foreground">
                          <User className="h-4 w-4" />
                        </div>
                        <Input
                          type="text"
                          name="full_name"
                          placeholder="Full Name"
                          className="pl-9 h-10 transition-shadow focus:ring-2 focus:ring-primary/20"
                          value={formData.full_name}
                          onChange={handleChange}
                        />
                      </div>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <label className="text-xs font-semibold text-foreground">Gender</label>
                        <Select
                          name="gender"
                          value={formData.gender}
                          onChange={handleChange}
                          className="h-10 transition-shadow focus:ring-2 focus:ring-primary/20 cursor-pointer"
                        >
                          <option value="">Select Gender</option>
                          <option value="Male">Male</option>
                          <option value="Female">Female</option>
                        </Select>
                      </div>

                      <div className="space-y-2">
                        <label className="text-xs font-semibold text-foreground">Date of Birth</label>
                        <div className="relative">
                          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-muted-foreground">
                            <Calendar className="h-4 w-4" />
                          </div>
                          <Input
                            type="date"
                            name="date_of_birth"
                            className="pl-9 h-10 transition-shadow focus:ring-2 focus:ring-primary/20 cursor-pointer"
                            value={formData.date_of_birth}
                            onChange={handleChange}
                          />
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Group 2: Contact Information */}
                  <div className="space-y-4">
                    <div className="flex items-center gap-2 border-b border-border pb-2 mb-4">
                      <Contact className="h-4 w-4 text-primary" />
                      <h3 className="text-sm font-bold uppercase tracking-wider text-muted-foreground">Contact Information</h3>
                    </div>

                    <div className="space-y-2">
                      <label className="text-xs font-semibold text-foreground">Email Address</label>
                      <div className="relative">
                        <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-muted-foreground">
                          <Mail className="h-4 w-4" />
                        </div>
                        <Input
                          type="email"
                          name="email"
                          placeholder="Email Address"
                          className="pl-9 h-10 transition-shadow focus:ring-2 focus:ring-primary/20"
                          value={formData.email}
                          onChange={handleChange}
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
                          placeholder="Phone Number"
                          className="pl-9 h-10 transition-shadow focus:ring-2 focus:ring-primary/20"
                          value={formData.phone}
                          onChange={handleChange}
                        />
                      </div>
                    </div>
                  </div>

                  {/* Validation & Feedback Area */}
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
                    </AnimatePresence>
                  </div>

                  <div className="pt-2">
                    <Button
                      type="submit"
                      disabled={loading}
                      className="w-full h-11 font-semibold text-sm cursor-pointer shadow-lg hover:shadow-xl hover:-translate-y-0.5 transition-all flex justify-center items-center gap-2"
                    >
                      {loading ? "Registering..." : "Continue to Diagnosis"}
                      {!loading && <ArrowRight className="h-4 w-4" />}
                    </Button>
                  </div>
                </form>
              </CardContent>
            </Card>
          </motion.div>
        </Container>
      </Section>
    </AppLayout>
  );
}

export default RegisterPage;