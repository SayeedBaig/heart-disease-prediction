import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import {
  LayoutDashboard, Users, Calendar, FileText, Activity, User, LogOut,
  Bell, Moon, Sun, UserCircle, Menu, ChevronRight, Stethoscope, Clock,
  CheckCircle2, AlertTriangle, ArrowRight, X, ShieldAlert, Search, FileBarChart, Zap, Check, Bot, ArrowLeft
} from "lucide-react";
import { Card, CardHeader, CardTitle, CardContent } from "../components/ui/Card";
import { Button } from "../components/ui/Button";
import { Grid } from "../components/ui/Grid";
import { Container } from "../components/ui/Container";
import { Section } from "../components/ui/Section";
import { EmptyState } from "../components/ui/EmptyState";

export default function DoctorDashboard({ doctor, onNavigate, onLogout, children, activeTab = "hub" }) {
  const navigate = useNavigate();
  const currentDoctor = doctor || JSON.parse(localStorage.getItem("cardio-doctor") || "{}");
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
    if (onNavigate) {
      onNavigate(key);
    } else if (route) {
      navigate(route);
    }
  };

  const handleLogoutAction = () => {
    if (onLogout) {
      onLogout();
    } else {
      localStorage.removeItem("doctor_access_token");
      localStorage.removeItem("cardio-doctor");
      navigate("/doctor/login");
    }
  };

  const doctorNameRaw = currentDoctor.full_name || currentDoctor.name || "Sarah Wilson";
  const doctorName = doctorNameRaw.replace(/^Dr\.\s*/i, "");
  const hospitalName = currentDoctor.hospital || "Apollo Hospitals";
  const specialization = currentDoctor.specialization || "Cardiology";

  const navItems = [
    { name: "Overview", icon: LayoutDashboard, key: "hub", action: () => handleAction("hub", "/doctor/dashboard") },
    { name: "Patient Registry", icon: Users, key: "patients", action: () => handleAction("patients", "/doctor/patients") },
    { name: "Consultations", icon: Calendar, key: "appointments", action: () => handleAction("appointments", "/doctor/appointments") },
    { name: "AI Reports", icon: FileBarChart, key: "reports", action: () => handleAction("reports", "/doctor/reports") },
    { name: "Profile Settings", icon: User, key: "profile", action: () => handleAction("profile", "/doctor/profile") },
  ];

  // Mock Data
  const stats = [
    { title: "Total Patients", value: "1,248", icon: Users, color: "text-blue-500", bg: "bg-blue-500/10" },
    { title: "Today's Appointments", value: "8", icon: Calendar, color: "text-purple-500", bg: "bg-purple-500/10" },
    { title: "Pending Reports", value: "14", icon: FileText, color: "text-warning", bg: "bg-warning/10" },
    { title: "Avg Risk Reviewed", value: "Moderate", icon: Activity, color: "text-success", bg: "bg-success/10" }
  ];

  const todaysAppointments = [
    { id: 1, patient: "Michael Johnson", time: "09:00 AM", type: "Initial Consultation", risk: "Unknown" },
    { id: 2, patient: "Emily Davis", time: "10:30 AM", type: "Follow-up", risk: "Low" },
    { id: 3, patient: "Robert Smith", time: "11:45 AM", type: "ECG Review", risk: "High" },
  ];

  const highRiskAlerts = [
    { id: "PT-092", patient: "Robert Smith", condition: "Elevated Troponin & ECG Abnormalities", time: "1 hour ago" },
    { id: "PT-104", patient: "James Wilson", condition: "Suspected CAD Progression", time: "3 hours ago" }
  ];

  const pendingReviews = [
    { id: "REP-440", patient: "Sarah Connor", scan: "Echocardiogram", date: "Oct 12" },
    { id: "REP-441", patient: "John Doe", scan: "ECG 12-lead", date: "Oct 12" }
  ];

  const activityTimeline = [
    { id: 1, action: "Approved Echocardiogram Report for PT-088", time: "10 mins ago", icon: CheckCircle2, color: "text-success" },
    { id: 2, action: "Updated clinical notes for Emily Davis", time: "1 hour ago", icon: FileText, color: "text-blue-500" },
    { id: 3, action: "Received critical alert for Robert Smith", time: "1 hour ago", icon: AlertTriangle, color: "text-destructive" },
  ];

  const [isLoading, setIsLoading] = useState(true);
  useEffect(() => {
    // Simulate loading state for skeleton demonstration
    const timer = setTimeout(() => setIsLoading(false), 800);
    return () => clearTimeout(timer);
  }, []);

  const DashboardSkeleton = () => (
    <div className="space-y-8 animate-pulse">
      <div className="h-40 rounded-2xl bg-secondary w-full"></div>
      <Grid cols={1} sm={2} lg={4} gap={4}>
        {[1, 2, 3, 4].map(i => (
          <div key={i} className="h-28 rounded-xl bg-secondary"></div>
        ))}
      </Grid>
      <div className="h-12 w-64 rounded-md bg-secondary"></div>
      <Grid cols={1} lg={3} gap={6}>
        <div className="lg:col-span-2 space-y-6">
          <div className="h-72 rounded-xl bg-secondary"></div>
          <div className="h-64 rounded-xl bg-secondary"></div>
        </div>
        <div className="space-y-6">
          <div className="h-48 rounded-xl bg-secondary"></div>
          <div className="h-56 rounded-xl bg-secondary"></div>
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
        className={`fixed md:relative z-50 w-[280px] h-full bg-background border-r border-border flex flex-col transition-transform duration-300 ease-in-out md:translate-x-0 ${sidebarOpen ? "translate-x-0" : "-translate-x-full"}`}
      >
        <div className="h-16 flex items-center px-6 border-b border-border">
          <div className="flex items-center gap-2 text-primary">
            <div className="h-8 w-8 rounded-lg bg-primary text-primary-foreground flex items-center justify-center shadow-sm">
              <Stethoscope className="h-5 w-5" />
            </div>
            <span className="font-extrabold text-lg tracking-tight text-foreground">CardioAI Pro</span>
          </div>
          <button onClick={() => setSidebarOpen(false)} className="md:hidden ml-auto text-muted-foreground hover:text-foreground">
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto py-6 px-4 space-y-1">
          <div className="text-xs font-bold text-muted-foreground uppercase tracking-wider mb-4 px-2">Clinical Workspace</div>
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.key;
            return (
              <button
                key={item.name}
                onClick={item.action}
                className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all group cursor-pointer ${
                  isActive 
                    ? "bg-primary text-primary-foreground shadow-md" 
                    : "text-muted-foreground hover:bg-secondary hover:text-foreground"
                }`}
              >
                <Icon className={`h-5 w-5 ${isActive ? "text-primary-foreground" : "text-muted-foreground group-hover:text-foreground transition-colors"}`} />
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
            <LogOut className="h-5 w-5 text-destructive group-hover:scale-110 transition-transform" />
            Sign Out Session
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
              <span>Clinician Portal</span>
              <ChevronRight className="h-4 w-4 mx-2" />
              <span className="text-foreground capitalize">{activeTab === "hub" ? "Dashboard" : activeTab}</span>
            </div>
          </div>

          <div className="flex items-center gap-2 md:gap-4">
            <div className="hidden md:flex relative group cursor-text">
              <Search className="h-4 w-4 absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
              <input 
                type="text" 
                placeholder="Search patients (Ctrl+K)" 
                className="h-9 w-64 pl-9 pr-4 rounded-full bg-secondary/50 border border-border text-sm focus:outline-none focus:ring-2 focus:ring-primary/20 transition-all"
              />
            </div>
            
            <Button variant="ghost" size="icon" className="h-9 w-9 rounded-full text-muted-foreground cursor-pointer relative">
              <Bell className="h-4 w-4" />
              <span className="absolute top-1 right-1 h-2 w-2 rounded-full bg-destructive border border-background"></span>
            </Button>
            <Button variant="ghost" size="icon" onClick={toggleTheme} className="h-9 w-9 rounded-full text-muted-foreground cursor-pointer">
              {theme === "light" ? <Moon className="h-4 w-4" /> : <Sun className="h-4 w-4" />}
            </Button>
            <div className="flex items-center gap-3 ml-2 pl-4 border-l border-border">
              <div className="hidden sm:block text-sm font-semibold text-foreground text-right leading-tight">
                Dr. {doctorName.split(" ")[0]}
                <div className="text-[10px] text-muted-foreground font-normal uppercase tracking-wider">{specialization}</div>
              </div>
              <button 
                onClick={() => handleAction("profile", "/doctor/profile")}
                className="h-9 w-9 rounded-full bg-primary/10 flex items-center justify-center border border-primary/20 text-primary cursor-pointer hover:bg-primary/20 transition-colors"
              >
                <UserCircle className="h-5 w-5" />
              </button>
            </div>
          </div>
        </header>

        {/* Scrollable Content Area */}
        <main className="flex-1 overflow-auto bg-background">
          {children ? (
            children
          ) : (
            <Section noPadding className="py-6 lg:py-8">
              <Container className="max-w-7xl space-y-8">
                
                {isLoading ? (
                  <DashboardSkeleton />
                ) : (
                  <motion.div 
                    initial={{ opacity: 0, y: 20 }} 
                    animate={{ opacity: 1, y: 0 }} 
                    transition={{ duration: 0.5 }}
                    className="space-y-8"
                  >
                    {/* 1. Doctor Welcome Banner */}
                    <div className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-primary to-blue-700 text-primary-foreground p-8 shadow-xl">
                      <div className="absolute top-0 right-0 p-8 opacity-10 pointer-events-none">
                        <Stethoscope className="w-64 h-64" />
                      </div>
                      <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-6">
                        <div>
                          <div className="flex items-center gap-2 mb-3">
                            <span className="px-2.5 py-0.5 rounded-full bg-white/20 text-white text-xs font-bold shadow-sm backdrop-blur-sm">
                              {hospitalName}
                            </span>
                            <span className="text-primary-foreground/80 text-sm font-medium">
                              {new Date().toLocaleDateString('en-US', { weekday: 'long', month: 'short', day: 'numeric' })}
                            </span>
                          </div>
                          <h1 className="text-3xl md:text-4xl font-bold mb-2 tracking-tight">
                            Welcome, Dr. {doctorName}
                          </h1>
                          <p className="text-primary-foreground/90 max-w-xl text-sm md:text-base">
                            You have <strong className="text-white">8 appointments</strong> today and <strong className="text-white">2 critical alerts</strong> requiring immediate attention.
                          </p>
                        </div>
                        <button 
                          onClick={() => handleAction("appointments", "/doctor/appointments")}
                          className="inline-flex items-center justify-center rounded-xl bg-white text-blue-700 hover:bg-blue-50 font-bold whitespace-nowrap shadow-lg self-start md:self-center cursor-pointer h-12 px-6 transition-colors"
                        >
                          <Calendar className="mr-2 h-5 w-5" /> View Schedule
                        </button>
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
                        <Button variant="outline" onClick={() => handleAction("patients", "/doctor/patients")} className="gap-2 shadow-sm bg-background border-border hover:bg-secondary cursor-pointer">
                          <Users className="h-4 w-4 text-blue-500" /> Patient Registry
                        </Button>
                        <Button variant="outline" onClick={() => handleAction("assistant")} className="gap-2 shadow-sm bg-background border-border hover:bg-secondary cursor-pointer">
                          <Zap className="h-4 w-4 text-warning" /> AI Prediction Review
                        </Button>
                        <Button variant="outline" className="gap-2 shadow-sm bg-background border-border hover:bg-secondary cursor-pointer">
                          <Activity className="h-4 w-4 text-success" /> Digital Twin Viewer
                        </Button>
                        <Button variant="outline" onClick={() => handleAction("reports", "/doctor/reports")} className="gap-2 shadow-sm bg-background border-border hover:bg-secondary cursor-pointer">
                          <FileBarChart className="h-4 w-4 text-purple-500" /> Clinic Reports
                        </Button>
                      </div>
                    </div>

                    <Grid cols={1} lg={3} gap={6}>
                      {/* Left Column */}
                      <div className="lg:col-span-2 space-y-6">
                        
                        {/* 4. Today's Appointments Table */}
                        <Card className="border-border bg-background shadow-sm overflow-hidden">
                          <CardHeader className="flex flex-row items-center justify-between pb-2 border-b border-border bg-secondary/20">
                            <CardTitle className="text-lg font-bold flex items-center gap-2">
                              <Calendar className="h-5 w-5 text-primary" /> Today's Schedule
                            </CardTitle>
                            <Button variant="ghost" size="sm" onClick={() => handleAction("appointments", "/doctor/appointments")} className="text-xs text-primary h-8 cursor-pointer">Manage</Button>
                          </CardHeader>
                          <div className="overflow-x-auto">
                            <table className="w-full text-sm text-left">
                              <thead className="bg-secondary/30 text-xs uppercase font-semibold text-muted-foreground border-b border-border">
                                <tr>
                                  <th className="px-6 py-3">Time</th>
                                  <th className="px-6 py-3">Patient</th>
                                  <th className="px-6 py-3">Type</th>
                                  <th className="px-6 py-3">AI Risk Score</th>
                                  <th className="px-6 py-3 text-right">Action</th>
                                </tr>
                              </thead>
                              <tbody className="divide-y divide-border">
                                {todaysAppointments.map((apt) => (
                                  <tr key={apt.id} className="hover:bg-secondary/20 transition-colors">
                                    <td className="px-6 py-4 font-medium text-foreground whitespace-nowrap">
                                      <div className="flex items-center gap-2">
                                        <Clock className="h-4 w-4 text-muted-foreground" />
                                        {apt.time}
                                      </div>
                                    </td>
                                    <td className="px-6 py-4 font-semibold text-foreground">{apt.patient}</td>
                                    <td className="px-6 py-4 text-muted-foreground">{apt.type}</td>
                                    <td className="px-6 py-4">
                                      <span className={`px-2 py-1 rounded-md text-xs font-semibold ${
                                        apt.risk === 'High' ? 'bg-destructive/10 text-destructive border border-destructive/20' :
                                        apt.risk === 'Low' ? 'bg-success/10 text-success border border-success/20' :
                                        'bg-secondary text-muted-foreground border border-border'
                                      }`}>
                                        {apt.risk}
                                      </span>
                                    </td>
                                    <td className="px-6 py-4 text-right">
                                      <Button variant="ghost" size="sm" className="h-8 text-xs cursor-pointer text-primary hover:bg-primary/10">View Chart</Button>
                                    </td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </Card>

                        {/* 7. Pending AI Reviews */}
                        <Card className="border-border bg-background shadow-sm">
                          <CardHeader className="flex flex-row items-center justify-between pb-2">
                            <CardTitle className="text-base font-bold flex items-center gap-2">
                              <CheckCircle2 className="h-5 w-5 text-warning" /> Pending AI Reviews
                            </CardTitle>
                          </CardHeader>
                          <CardContent>
                            <div className="space-y-3 mt-2">
                              {pendingReviews.map((rev, i) => (
                                <div key={i} className="flex items-center justify-between p-4 rounded-xl border border-border bg-secondary/10 hover:bg-secondary/30 transition-colors group">
                                  <div className="flex items-center gap-4">
                                    <div className="h-10 w-10 rounded-lg bg-warning/10 flex items-center justify-center text-warning">
                                      <Activity className="h-5 w-5" />
                                    </div>
                                    <div>
                                      <p className="text-sm font-semibold text-foreground">{rev.scan} • {rev.patient}</p>
                                      <p className="text-xs text-muted-foreground">Generated on {rev.date} • Needs Verification</p>
                                    </div>
                                  </div>
                                  <Button variant="outline" size="sm" className="h-8 text-xs cursor-pointer bg-background">Review</Button>
                                </div>
                              ))}
                            </div>
                          </CardContent>
                        </Card>
                      </div>

                      {/* Right Column */}
                      <div className="space-y-6">
                        {/* 6. High Risk Alerts */}
                        <Card className="border-destructive/20 shadow-sm relative overflow-hidden bg-destructive/5">
                          <div className="absolute top-0 right-0 w-2 h-full bg-destructive"></div>
                          <CardHeader className="pb-2">
                            <CardTitle className="text-base font-bold flex items-center gap-2 text-destructive">
                              <ShieldAlert className="h-5 w-5" /> Critical Alerts
                            </CardTitle>
                          </CardHeader>
                          <CardContent>
                            <div className="space-y-3 mt-2">
                              {highRiskAlerts.map((alert, i) => (
                                <div key={i} className="p-3 rounded-lg bg-background border border-destructive/20 shadow-sm relative">
                                  <div className="flex justify-between items-start mb-1">
                                    <h4 className="font-bold text-foreground text-sm">{alert.patient}</h4>
                                    <span className="text-[10px] text-muted-foreground font-medium">{alert.time}</span>
                                  </div>
                                  <p className="text-xs text-destructive font-medium leading-relaxed">{alert.condition}</p>
                                  <Button variant="ghost" size="sm" className="w-full mt-2 h-7 text-xs bg-destructive/10 text-destructive hover:bg-destructive hover:text-white cursor-pointer">
                                    Take Action
                                  </Button>
                                </div>
                              ))}
                            </div>
                          </CardContent>
                        </Card>

                        {/* 8. AI Clinical Insights */}
                        <Card className="border-border shadow-sm bg-gradient-to-br from-background to-secondary/40">
                          <CardHeader className="pb-2">
                            <CardTitle className="text-base font-bold flex items-center gap-2">
                              <Bot className="h-5 w-5 text-purple-500" /> AI Insights
                            </CardTitle>
                          </CardHeader>
                          <CardContent>
                            <p className="text-sm text-muted-foreground leading-relaxed mb-4">
                              Machine learning models indicate a <strong className="text-foreground">12% increase</strong> in CAD progression indicators among males 50-60 in your registry this month.
                            </p>
                            <Button variant="outline" onClick={() => handleAction("assistant")} className="w-full text-xs h-8 cursor-pointer border-purple-500/20 text-purple-500 hover:bg-purple-500/10">
                              View Full Analysis
                            </Button>
                          </CardContent>
                        </Card>

                        {/* 9. Activity Timeline */}
                        <Card className="border-border shadow-sm">
                          <CardHeader className="pb-3">
                            <CardTitle className="text-base font-bold">Recent Activity</CardTitle>
                          </CardHeader>
                          <CardContent>
                            <div className="relative pl-6 mt-2 space-y-6 before:absolute before:inset-y-0 before:left-[11px] before:w-[2px] before:bg-border">
                              {activityTimeline.map((item, i) => (
                                <div key={item.id} className="relative">
                                  <div className="absolute -left-[29px] h-6 w-6 rounded-full bg-background border-2 border-secondary flex items-center justify-center shadow-sm">
                                    <item.icon className={`h-3 w-3 ${item.color}`} />
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
              </Container>
            </Section>
          )}
        </main>
      </div>
    </div>
  );
}
