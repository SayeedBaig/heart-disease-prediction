import { useNavigate } from "react-router-dom";
import { Stethoscope, HeartPulse, LogIn, UserPlus, CheckCircle2 } from "lucide-react";
import { AppLayout } from "../components/ui/AppLayout";
import { Container } from "../components/ui/Container";
import { Section } from "../components/ui/Section";
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from "../components/ui/Card";
import { Button } from "../components/ui/Button";
import { Grid } from "../components/ui/Grid";

export default function RoleSelection() {
  const navigate = useNavigate();

  const doctorFeatures = [
    "Manage patient cohorts securely",
    "Analyze ECG & Echo diagnostic scans",
    "Issue signed clinical assessments",
    "Review comprehensive AI risk reports"
  ];

  const patientFeatures = [
    "Evaluate heart disease risk profiles",
    "Explore interactive Digital Twin simulations",
    "View detailed AI clinical reports",
    "Book consultations with specialists"
  ];

  return (
    <AppLayout>
      <Section className="min-h-[85vh] flex items-center justify-center bg-secondary/10">
        <Container>
          <div className="text-center max-w-2xl mx-auto mb-16 mt-8 md:mt-0">
            <h1 className="text-4xl font-extrabold tracking-tight sm:text-5xl text-foreground">
              Choose Your Workspace
            </h1>
            <p className="mt-5 text-lg text-muted-foreground leading-relaxed">
              Welcome to CardioAI. Choose your portal below to sign in or register for a new account.
            </p>
          </div>

          <div className="max-w-5xl mx-auto">
            <Grid cols={2} gap={8}>
              {/* Doctor Portal Card */}
              <Card className="hover:shadow-lg transition-all hover:border-primary/40 bg-background">
                <CardHeader>
                  <div className="h-16 w-16 rounded-2xl bg-primary/10 flex items-center justify-center mb-6 border border-primary/20">
                    <Stethoscope className="h-8 w-8 text-primary" />
                  </div>
                  <CardTitle className="text-3xl font-bold">Doctor Portal</CardTitle>
                  <CardDescription className="text-base mt-2 leading-relaxed">
                    Comprehensive clinical workspace for medical practitioners.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-4 mt-2">
                    {doctorFeatures.map(f => (
                      <li key={f} className="flex items-start gap-3">
                        <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                        <span className="text-sm font-medium text-foreground">{f}</span>
                      </li>
                    ))}
                  </ul>
                </CardContent>
                <CardFooter className="flex flex-col gap-3 pt-8 border-t border-border mt-auto">
                  <Button 
                    size="lg"
                    className="w-full h-12 font-bold text-sm" 
                    onClick={() => navigate("/doctor/login")}
                  >
                    <LogIn className="mr-2 h-5 w-5" />
                    Doctor Login
                  </Button>
                  <Button 
                    variant="outline" 
                    size="lg"
                    className="w-full h-12 font-bold text-sm" 
                    onClick={() => navigate("/doctor/register")}
                  >
                    <UserPlus className="mr-2 h-5 w-5" />
                    Doctor Sign Up
                  </Button>
                </CardFooter>
              </Card>

              {/* Patient Portal Card */}
              <Card className="hover:shadow-lg transition-all hover:border-primary/40 bg-background">
                <CardHeader>
                  <div className="h-16 w-16 rounded-2xl bg-primary/10 flex items-center justify-center mb-6 border border-primary/20">
                    <HeartPulse className="h-8 w-8 text-primary" />
                  </div>
                  <CardTitle className="text-3xl font-bold">Patient Portal</CardTitle>
                  <CardDescription className="text-base mt-2 leading-relaxed">
                    Personalized cardiovascular health suite for patients.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-4 mt-2">
                    {patientFeatures.map(f => (
                      <li key={f} className="flex items-start gap-3">
                        <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                        <span className="text-sm font-medium text-foreground">{f}</span>
                      </li>
                    ))}
                  </ul>
                </CardContent>
                <CardFooter className="flex flex-col gap-3 pt-8 border-t border-border mt-auto">
                  <Button 
                    size="lg"
                    className="w-full h-12 font-bold text-sm" 
                    onClick={() => navigate("/patient/login")}
                  >
                    <LogIn className="mr-2 h-5 w-5" />
                    Patient Login
                  </Button>
                  <Button 
                    variant="outline" 
                    size="lg"
                    className="w-full h-12 font-bold text-sm" 
                    onClick={() => navigate("/patient/signup")}
                  >
                    <UserPlus className="mr-2 h-5 w-5" />
                    Patient Sign Up
                  </Button>
                </CardFooter>
              </Card>
            </Grid>
          </div>
        </Container>
      </Section>
    </AppLayout>
  );
}
