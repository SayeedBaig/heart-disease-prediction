import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import {
  LayoutDashboard, HeartPulse, Activity, Bot, FileText, Calendar,
  Settings, LogOut, Bell, Moon, Sun, UserCircle, Menu, ChevronRight,
  Stethoscope, Clock, CheckCircle2, Plus, PlayCircle, X, ArrowLeft
} from "lucide-react";
import { Card, CardHeader, CardTitle, CardContent } from "../components/ui/Card";
import { Button } from "../components/ui/Button";
import { Grid } from "../components/ui/Grid";
import { Container } from "../components/ui/Container";
import { Section } from "../components/ui/Section";
import { EmptyState } from "../components/ui/EmptyState";

export default function PatientDashboard({ onNavigate, onLogout, children, activeTab = "hub" }) {
  const navigate = useNavigate();
  const patient = JSON.parse(localStorage.getItem("cardio-patient") || "{}");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  
  // Theme management
  const [theme, setTheme] = useState(() => {
    if (typeof window !== "undefined") return localStorage.getItem("theme") || "light";
    return "light";
  });

  useEffect(() => {
    if (theme === "dark") document.documentElement.classList.add("dark");
    else document.documentElement.classList.remove("dark");
    localStorage.setItem("theme", theme);
  }, [theme]);

  const toggleTheme = () => setTheme(prev => prev === "light" ? "dark" : "light");

  const handleAction = (key, route) => {
    setSidebarOpen(false);
    if (key === "assistant") {
      window.dispatchEvent(new Event("open-chatbot"));
      return;
    }
    if (onNavigate) onNavigate(key);
    if (route) navigate(route);
  };

  const handleLogoutAction = () => {
    if (onLogout) onLogout();
    else {
      localStorage.removeItem("access_token");
      localStorage.removeItem("cardio-patient");
      navigate("/");
    }
  };

  const navItems = [
    { name: "Dashboard", icon: LayoutDashboard, key: "hub", action: () => handleAction("hub", "/patient/dashboard") },
    { name: "Heart Prediction", icon: HeartPulse, key: "intake", action: () => handleAction("intake", "/patient/intake") },
    { name: "Digital Twin", icon: Activity, key: "twin", action: () => handleAction("twin", "/patient/digital-twin") },
    { name: "AI Assistant", icon: Bot, key: "assistant", action: () => handleAction("assistant") },
    { name: "Reports", icon: FileText, key: "report", action: () => handleAction("report", "/patient/reports") },
    { name: "Appointments", icon: Calendar, key: "appointment", action: () => handleAction("appointment", "/patient/appointments") },
    { name: "Settings", icon: Settings, key: "profile", action: () => handleAction("profile", "/patient/profile") },
  ];

  // Mock data for dashboard
  const stats = [
    { title: "Total Predictions", value: "3", icon: HeartPulse, color: "text-blue-500", bg: "bg-blue-500/10" },
    { title: "Current Risk", value: "Low", icon: CheckCircle2, color: "text-success", bg: "bg-success/10" },
    { title: "Reports Generated", value: "12", icon: FileText, color: "text-purple-500", bg: "bg-purple-500/10" },
    { title: "Digital Twin", value: "Active", icon: Activity, color: "text-warning", bg: "bg-warning/10" }
  ];

  const recentPredictions = [
    { id: "PRD-001", date: "Oct 12, 2023", risk: "Low", status: "Completed" },
    { id: "PRD-002", date: "Sep 28, 2023", risk: "Low", status: "Completed" }
  ];

  const recentReports = [
    { id: "REP-992", name: "Comprehensive Cardiac Panel", date: "Oct 13, 2023" },
    { id: "REP-991", name: "Echocardiogram Summary", date: "Oct 01, 2023" }
  ];

  const upcomingAppointment = {
    doctor: "Dr. Sarah Jenkins",
    specialty: "Cardiologist",
    date: "Nov 05, 2023",
    time: "10:30 AM",
    type: "Video Consultation"
  };

  const activityTimeline = [
    { id: 1, action: "Viewed Digital Twin", time: "2 hours ago", icon: Activity },
    { id: 2, action: "Downloaded Report REP-992", time: "1 day ago", icon: FileText },
    { id: 3, action: "Completed AI Screening", time: "Oct 12, 2023", icon: HeartPulse }
  ];

  const [isLoading, setIsLoading] = useState(true);
  useEffect(() => {
    // Simulate loading state for skeleton demonstration
    const timer = setTimeout(() => setIsLoading(false), 800);
    return () => clearTimeout(timer);
  }, []);

  const DashboardSkeleton = () => (
    <div className="space-y-8 animate-pulse">
      <div className="h-48 rounded-2xl bg-secondary w-full"></div>
      <Grid cols={1} sm={2} lg={4} gap={4}>
        {[1, 2, 3, 4].map(i => (
          <div key={i} className="h-28 rounded-xl bg-secondary"></div>
        ))}
      </Grid>
      <div className="h-10 w-48 rounded-md bg-secondary"></div>
      <Grid cols={1} lg={3} gap={6}>
        <div className="lg:col-span-2 space-y-6">
          <div className="h-80 rounded-xl bg-secondary"></div>
          <div className="h-64 rounded-xl bg-secondary"></div>
        </div>
        <div className="space-y-6">
          <div className="h-56 rounded-xl bg-secondary"></div>
          <div className="h-48 rounded-xl bg-secondary"></div>
          <div className="h-64 rounded-xl bg-secondary"></div>
        </div>
      </Grid>
    </div>
  );

  return (
    <div className="flex h-screen bg-secondary/30 overflow-hidden font-sans text-foreground">
      {/* Mobile Sidebar Overlay */}
      <AnimatePresence>
        {sidebarOpen && (
          <motion.div 
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            onClick={() => setSidebarOpen(false)}
            className="fixed inset-0 bg-background/80 backdrop-blur-sm z-40 md:hidden"
          />
        )}
      </AnimatePresence>

      {/* Fixed Collapsible Left Sidebar */}
      <motion.aside 
        className={`fixed md:relative z-50 w-[260px] h-full bg-background border-r border-border flex flex-col transition-transform duration-300 ease-in-out md:translate-x-0 ${sidebarOpen ? "translate-x-0" : "-translate-x-full"}`}
      >
        <div className="h-16 flex items-center px-6 border-b border-border">
          <div className="flex items-center gap-2 text-primary">
            <div className="h-8 w-8 rounded-lg bg-primary/10 border border-primary/20 text-primary flex items-center justify-center shadow-sm">
              <Stethoscope className="h-5 w-5" />
            </div>
            <span className="font-bold text-lg tracking-tight text-foreground">CardioAI</span>
          </div>
          <button onClick={() => setSidebarOpen(false)} className="md:hidden ml-auto text-muted-foreground hover:text-foreground">
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto py-6 px-4 space-y-1">
          <div className="text-xs font-bold text-muted-foreground uppercase tracking-wider mb-4 px-2">Main Menu</div>
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.key;
            return (
              <button
                key={item.name}
                onClick={item.action}
                className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all group cursor-pointer ${
                  isActive 
                    ? "bg-primary/10 text-primary font-semibold" 
                    : "text-muted-foreground hover:bg-secondary hover:text-foreground"
                }`}
              >
                <Icon className={`h-4 w-4 ${isActive ? "text-primary" : "text-muted-foreground group-hover:text-foreground transition-colors"}`} />
                {item.name}
              </button>
            );
          })}
        </div>

        <div className="p-4 border-t border-border">
          <button
            onClick={handleLogoutAction}
            className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium text-destructive hover:bg-destructive/10 transition-colors cursor-pointer group"
          >
            <LogOut className="h-4 w-4 text-destructive group-hover:scale-110 transition-transform" />
            Logout
          </button>
        </div>
      </motion.aside>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Sticky Top Header */}
        <header className="h-16 bg-background/80 backdrop-blur-md sticky top-0 border-b border-border flex items-center justify-between px-4 md:px-8 z-30">
          <div className="flex items-center gap-4">
            <button 
              onClick={() => setSidebarOpen(true)}
              className="md:hidden p-2 rounded-lg text-muted-foreground hover:bg-secondary transition-colors"
            >
              <Menu className="h-5 w-5" />
            </button>
            
            <button
              onClick={() => navigate(-1)}
              className="hidden md:flex items-center gap-1 text-sm font-medium text-muted-foreground hover:text-foreground transition-colors mr-2 cursor-pointer"
            >
              <ArrowLeft className="h-4 w-4" /> Back
            </button>
            <div className="hidden md:flex items-center text-sm font-medium text-muted-foreground border-l border-border pl-4">
              <span>Patient Portal</span>
              <ChevronRight className="h-4 w-4 mx-2" />
              <span className="text-foreground capitalize">{activeTab === "hub" ? "Dashboard" : activeTab === "intake" ? "Prediction" : activeTab}</span>
            </div>
          </div>

          <div className="flex items-center gap-2 md:gap-4">
            <Button variant="ghost" size="icon" className="h-9 w-9 rounded-full text-muted-foreground cursor-pointer">
              <Bell className="h-4 w-4" />
            </Button>
            <Button variant="ghost" size="icon" onClick={toggleTheme} className="h-9 w-9 rounded-full text-muted-foreground cursor-pointer">
              {theme === "light" ? <Moon className="h-4 w-4" /> : <Sun className="h-4 w-4" />}
            </Button>
            <div className="flex items-center gap-3 ml-2 pl-4 border-l border-border">
              <div className="hidden sm:block text-sm font-semibold text-foreground">
                {patient.full_name?.split(" ")[0] || patient.name || "Patient"}
              </div>
              <button 
                onClick={() => handleAction("profile", "/patient/profile")}
                className="h-9 w-9 rounded-full bg-primary/10 flex items-center justify-center border border-primary/20 text-primary cursor-pointer hover:bg-primary/20 transition-colors"
              >
                <UserCircle className="h-5 w-5" />
              </button>
            </div>
          </div>
        </header>

        {/* Scrollable Content Area */}
        <main className="flex-1 overflow-auto bg-background">
          <Section noPadding className="py-6 lg:py-8">
            <Container className="max-w-7xl space-y-8">
              {children ? (
                children
              ) : (
                <>
                  {isLoading ? (
                    <DashboardSkeleton />
                  ) : (
                    <motion.div 
                      initial={{ opacity: 0, y: 20 }} 
                      animate={{ opacity: 1, y: 0 }} 
                      transition={{ duration: 0.5 }}
                      className="space-y-8"
                    >
                      {/* 1. Welcome Banner */}
                    <div className="relative overflow-hidden rounded-2xl bg-primary text-primary-foreground p-8 shadow-xl">
                      <div className="absolute top-0 right-0 p-8 opacity-10 pointer-events-none">
                        <HeartPulse className="w-64 h-64" />
                      </div>
                      <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-6">
                        <div>
                          <div className="flex items-center gap-2 mb-3">
                            <span className="text-primary-foreground/80 text-sm font-medium">
                              {new Date().toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}
                            </span>
                            <span className="px-2.5 py-0.5 rounded-full bg-success text-success-foreground text-xs font-bold shadow-sm">
                              Low Risk
                            </span>
                          </div>
                          <h1 className="text-3xl md:text-4xl font-bold mb-2 tracking-tight">
                            Good morning, {patient.full_name?.split(" ")[0] || patient.name || "Patient"}
                          </h1>
                          <p className="text-primary-foreground/90 max-w-xl text-sm md:text-base">
                            Your cardiovascular health is stable. Ready for your next AI-powered screening?
                          </p>
                        </div>
                        <Button 
                          onClick={() => handleAction("intake", "/patient/intake")}
                          className="bg-background text-primary hover:bg-secondary whitespace-nowrap shadow-lg self-start md:self-center cursor-pointer h-12 px-6"
                        >
                          <PlayCircle className="mr-2 h-5 w-5" /> Start Prediction
                        </Button>
                      </div>
                    </div>

                    {/* 2. Statistics Cards */}
                    <Grid cols={1} sm={2} lg={4} gap={4}>
                      {stats.map((stat, i) => (
                        <Card key={i} className="border-border hover:shadow-lg hover:-translate-y-1 transition-all duration-300">
                          <CardContent className="p-6 flex items-center gap-4">
                            <div className={`h-12 w-12 rounded-xl ${stat.bg} flex items-center justify-center shrink-0`}>
                              <stat.icon className={`h-6 w-6 ${stat.color}`} />
                            </div>
                            <div>
                              <p className="text-sm font-medium text-muted-foreground">{stat.title}</p>
                              <h3 className="text-2xl font-bold text-foreground mt-1">{stat.value}</h3>
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                    </Grid>

                    {/* 3. Quick Actions */}
                    <div>
                      <h3 className="text-sm font-bold uppercase tracking-wider text-muted-foreground mb-4">Quick Actions</h3>
                      <div className="flex flex-wrap gap-3">
                        <Button variant="outline" onClick={() => handleAction("intake", "/patient/intake")} className="gap-2 shadow-sm bg-background border-border hover:bg-secondary cursor-pointer">
                          <Plus className="h-4 w-4 text-primary" /> New Prediction
                        </Button>
                        <Button variant="outline" onClick={() => handleAction("twin", "/patient/digital-twin")} className="gap-2 shadow-sm bg-background border-border hover:bg-secondary cursor-pointer">
                          <Activity className="h-4 w-4 text-warning" /> Digital Twin
                        </Button>
                        <Button variant="outline" onClick={() => handleAction("assistant")} className="gap-2 shadow-sm bg-background border-border hover:bg-secondary cursor-pointer">
                          <Bot className="h-4 w-4 text-purple-500" /> AI Assistant
                        </Button>
                        <Button variant="outline" onClick={() => handleAction("report", "/patient/reports")} className="gap-2 shadow-sm bg-background border-border hover:bg-secondary cursor-pointer">
                          <FileText className="h-4 w-4 text-blue-500" /> View Reports
                        </Button>
                      </div>
                    </div>

                    <Grid cols={1} lg={3} gap={6}>
                      {/* Left Column (Predictions & Reports) */}
                      <div className="lg:col-span-2 space-y-6">
                        {/* 4. Recent Predictions */}
                        <Card className="h-full border-border bg-background shadow-sm">
                          <CardHeader className="flex flex-row items-center justify-between pb-2">
                            <CardTitle className="text-lg font-bold">Recent Predictions</CardTitle>
                            <Button variant="ghost" size="sm" onClick={() => handleAction("intake", "/patient/intake")} className="text-xs text-primary h-8 cursor-pointer">View All</Button>
                          </CardHeader>
                          <CardContent>
                            {recentPredictions.length > 0 ? (
                              <div className="space-y-4 mt-2">
                                {recentPredictions.map((pred, i) => (
                                  <div key={i} className="flex items-center justify-between p-4 rounded-xl border border-border bg-secondary/30 hover:bg-secondary/50 transition-colors group cursor-pointer">
                                    <div className="flex items-center gap-4">
                                      <div className="h-10 w-10 rounded-xl bg-blue-500/10 flex items-center justify-center text-blue-500 border border-blue-500/20">
                                        <HeartPulse className="h-5 w-5" />
                                      </div>
                                      <div>
                                        <h4 className="text-sm font-semibold text-foreground group-hover:text-primary transition-colors">{pred.id}</h4>
                                        <p className="text-xs text-muted-foreground">{pred.date}</p>
                                      </div>
                                    </div>
                                    <div className="flex items-center gap-4">
                                      <span className="px-2.5 py-1 rounded-md bg-success/10 text-success text-xs font-semibold border border-success/20">
                                        {pred.risk} Risk
                                      </span>
                                      <ChevronRight className="h-4 w-4 text-muted-foreground group-hover:text-foreground transition-colors" />
                                    </div>
                                  </div>
                                ))}
                              </div>
                            ) : (
                              <EmptyState 
                                icon={<HeartPulse />}
                                title="No predictions yet"
                                description="Start your first AI heart risk screening today."
                                actionText="Start Prediction"
                                onAction={() => handleAction("intake", "/patient/intake")}
                              />
                            )}
                          </CardContent>
                        </Card>

                        {/* 5. Recent Reports */}
                        <Card className="border-border bg-background shadow-sm">
                          <CardHeader className="flex flex-row items-center justify-between pb-2">
                            <CardTitle className="text-lg font-bold">Recent Reports</CardTitle>
                            <Button variant="ghost" size="sm" onClick={() => handleAction("report", "/patient/reports")} className="text-xs text-primary h-8 cursor-pointer">View All</Button>
                          </CardHeader>
                          <CardContent>
                            <div className="space-y-1 mt-2">
                              {recentReports.map((rep, i) => (
                                <div key={i} className="flex items-center justify-between p-3 rounded-lg hover:bg-secondary/50 transition-colors group">
                                  <div className="flex items-center gap-3">
                                    <div className="h-8 w-8 rounded-lg bg-secondary flex items-center justify-center text-muted-foreground group-hover:text-primary transition-colors">
                                      <FileText className="h-4 w-4" />
                                    </div>
                                    <div>
                                      <p className="text-sm font-medium text-foreground">{rep.name}</p>
                                      <p className="text-xs text-muted-foreground">{rep.id} • {rep.date}</p>
                                    </div>
                                  </div>
                                  <Button variant="ghost" size="sm" className="h-8 text-xs cursor-pointer opacity-0 group-hover:opacity-100 transition-opacity">Download</Button>
                                </div>
                              ))}
                            </div>
                          </CardContent>
                        </Card>
                      </div>

                      {/* Right Column (Appointments, Tips, Timeline) */}
                      <div className="space-y-6">
                        {/* 6. Upcoming Appointment */}
                        <Card className="border-border overflow-hidden shadow-sm relative">
                          <div className="absolute top-0 right-0 w-32 h-32 bg-primary/5 rounded-bl-full -mr-16 -mt-16 pointer-events-none" />
                          <CardHeader className="pb-3">
                            <CardTitle className="text-base font-semibold flex items-center gap-2 text-foreground">
                              <Calendar className="h-4 w-4 text-primary" /> Upcoming Appointment
                            </CardTitle>
                          </CardHeader>
                          <CardContent>
                            <div className="p-4 rounded-xl border border-primary/20 bg-primary/5">
                              <h4 className="font-bold text-foreground text-sm">{upcomingAppointment.doctor}</h4>
                              <p className="text-xs text-primary font-medium mb-4">{upcomingAppointment.specialty}</p>
                              
                              <div className="flex items-center gap-3 text-sm mb-5 bg-background/80 p-2.5 rounded-lg border border-border">
                                <Clock className="h-4 w-4 text-muted-foreground" />
                                <div>
                                  <div className="font-semibold text-foreground">{upcomingAppointment.date}</div>
                                  <div className="text-xs text-muted-foreground">{upcomingAppointment.time}</div>
                                </div>
                              </div>
                              <Button className="w-full h-10 text-sm shadow-md cursor-pointer">
                                Join Video Call
                              </Button>
                            </div>
                          </CardContent>
                        </Card>

                        {/* 7. AI Health Tips */}
                        <Card className="border-border shadow-sm">
                          <CardHeader className="pb-2">
                            <CardTitle className="text-base font-semibold flex items-center gap-2">
                              <Bot className="h-4 w-4 text-purple-500" /> AI Health Insight
                            </CardTitle>
                          </CardHeader>
                          <CardContent>
                            <div className="p-4 rounded-xl bg-secondary/40 text-sm text-muted-foreground leading-relaxed">
                              Based on your last screening, maintaining a daily 30-minute walk can further reduce cardiovascular risks by up to <strong className="text-foreground">15%</strong>. Keep up the good work!
                            </div>
                          </CardContent>
                        </Card>

                        {/* 8. Activity Timeline */}
                        <Card className="border-border shadow-sm">
                          <CardHeader className="pb-3">
                            <CardTitle className="text-base font-semibold">Recent Activity</CardTitle>
                          </CardHeader>
                          <CardContent>
                            <div className="relative pl-6 mt-2 space-y-6 before:absolute before:inset-y-0 before:left-[11px] before:w-[2px] before:bg-border">
                              {activityTimeline.map((item, i) => (
                                <div key={item.id} className="relative">
                                  <div className="absolute -left-[29px] h-6 w-6 rounded-full bg-background border-2 border-primary/40 flex items-center justify-center shadow-sm">
                                    <item.icon className="h-3 w-3 text-primary" />
                                  </div>
                                  <div>
                                    <p className="text-sm font-medium text-foreground">{item.action}</p>
                                    <p className="text-xs text-muted-foreground mt-0.5">{item.time}</p>
                                  </div>
                                </div>
                              ))}
                            </div>
                          </CardContent>
                        </Card>
                      </div>
                    </Grid>

                  </motion.div>
                  )}
                </>
              )}
            </Container>
          </Section>
        </main>
      </div>
    </div>
  );
}
