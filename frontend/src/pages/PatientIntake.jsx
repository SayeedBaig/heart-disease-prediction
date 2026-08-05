import { useState, useRef } from "react";
import { motion } from "framer-motion";
import { 
  HeartPulse, 
  UploadCloud, 
  Activity, 
  AlertCircle, 
  CheckCircle2, 
  FileText,
  ScanHeart,
  ChevronRight,
  Clock,
  ShieldCheck,
  BrainCircuit,
  Zap,
  Trash2
} from "lucide-react";
import PatientDashboard from "./PatientDashboard";
import { Card, CardHeader, CardTitle, CardContent } from "../components/ui/Card";
import { Button } from "../components/ui/Button";
import { Input } from "../components/ui/Input";
import { Select } from "../components/ui/Select";
import { Grid } from "../components/ui/Grid";

export default function PatientIntake({ data, files, update, setFile, startAnalysis, isProcessing, progress, onNavigate, onLogout, patient }) {
  
  const ecgInputRef = useRef(null);
  const echoInputRef = useRef(null);

  const [dragActiveEcg, setDragActiveEcg] = useState(false);
  const [dragActiveEcho, setDragActiveEcho] = useState(false);

  const handleDrag = (e, setDrag) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") setDrag(true);
    else if (e.type === "dragleave") setDrag(false);
  };

  const handleDrop = (e, key, setDrag) => {
    e.preventDefault();
    e.stopPropagation();
    setDrag(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(key, e.dataTransfer.files[0]);
    }
  };

  // Check required fields (age, height, weight, systolic, diastolic)
  const isReady = data.age && data.height && data.weight && data.systolic && data.diastolic;

  return (
    <>
      {/* Top Row */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 mb-6">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-foreground">Heart Prediction Workspace</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Input clinical parameters and diagnostic files for AI assessment.
          </p>
        </div>
        <Button 
          onClick={() => {
            // clear form
            Object.keys(data).forEach(k => update(k, ""));
            setFile("ecg", null);
            setFile("echo", null);
          }}
          variant="outline" 
          className="bg-background cursor-pointer"
        >
          Clear Workspace
        </Button>
      </div>

      <div className="flex flex-col lg:flex-row gap-8">
        
        {/* LEFT COLUMN (65%) */}
        <div className="w-full lg:w-[65%] space-y-6">
          
          {/* Patient Information */}
          <Card className="border-border shadow-sm" noPadding>
            <CardHeader className="px-6 py-4 border-b border-border bg-secondary/30 mb-0">
              <div className="flex items-center gap-2 text-primary">
                <UserIcon className="h-5 w-5" />
                <CardTitle className="text-sm font-bold uppercase tracking-wider text-muted-foreground">Patient Information</CardTitle>
              </div>
            </CardHeader>
            <CardContent className="p-6">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="space-y-2">
                  <label className="text-xs font-semibold">Age (yrs) <span className="text-destructive">*</span></label>
                  <Input type="number" placeholder="e.g. 45" value={data.age} onChange={e => update("age", e.target.value)} />
                </div>
                <div className="space-y-2">
                  <label className="text-xs font-semibold">Height (cm) <span className="text-destructive">*</span></label>
                  <Input type="number" placeholder="e.g. 175" value={data.height} onChange={e => update("height", e.target.value)} />
                </div>
                <div className="space-y-2">
                  <label className="text-xs font-semibold">Weight (kg) <span className="text-destructive">*</span></label>
                  <Input type="number" placeholder="e.g. 70" value={data.weight} onChange={e => update("weight", e.target.value)} />
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Clinical Parameters */}
          <Card className="border-border shadow-sm" noPadding>
            <CardHeader className="px-6 py-4 border-b border-border bg-secondary/30 mb-0">
              <div className="flex items-center gap-2 text-primary">
                <Activity className="h-5 w-5" />
                <CardTitle className="text-sm font-bold uppercase tracking-wider text-muted-foreground">Clinical Parameters</CardTitle>
              </div>
            </CardHeader>
            <CardContent className="p-6 space-y-6">
              
              {/* Blood Pressure */}
              <div>
                <h4 className="text-sm font-semibold mb-3 flex items-center gap-2">
                  <HeartPulse className="h-4 w-4 text-rose-500" /> Blood Pressure
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-2">
                    <label className="text-xs font-semibold">Systolic (mmHg) <span className="text-destructive">*</span></label>
                    <Input type="number" placeholder="e.g. 120" value={data.systolic} onChange={e => update("systolic", e.target.value)} />
                  </div>
                  <div className="space-y-2">
                    <label className="text-xs font-semibold">Diastolic (mmHg) <span className="text-destructive">*</span></label>
                    <Input type="number" placeholder="e.g. 80" value={data.diastolic} onChange={e => update("diastolic", e.target.value)} />
                  </div>
                </div>
              </div>

              <div className="w-full h-px bg-border" />

              {/* Blood Levels */}
              <div>
                <h4 className="text-sm font-semibold mb-3 flex items-center gap-2">
                  <DropletIcon className="h-4 w-4 text-blue-500" /> Blood Markers
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-2">
                    <label className="text-xs font-semibold">Cholesterol Level</label>
                    <Select value={data.cholesterol} onChange={e => update("cholesterol", e.target.value)} className="cursor-pointer">
                      <option value="">Select Level</option>
                      <option value="1">Normal (&lt; 200 mg/dL)</option>
                      <option value="2">Above Normal (200 - 239 mg/dL)</option>
                      <option value="3">High (&ge; 240 mg/dL)</option>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <label className="text-xs font-semibold">Fasting Glucose</label>
                    <Select value={data.glucose} onChange={e => update("glucose", e.target.value)} className="cursor-pointer">
                      <option value="">Select Level</option>
                      <option value="1">Normal (&lt; 100 mg/dL)</option>
                      <option value="2">Above Normal (100 - 125 mg/dL)</option>
                      <option value="3">High (&ge; 126 mg/dL)</option>
                    </Select>
                  </div>
                </div>
              </div>

              <div className="w-full h-px bg-border" />

              {/* Lifestyle */}
              <div>
                <h4 className="text-sm font-semibold mb-3 flex items-center gap-2">
                  <FlameIcon className="h-4 w-4 text-orange-500" /> Lifestyle Factors
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="space-y-2">
                    <label className="text-xs font-semibold">Smoking Status</label>
                    <Select value={data.smoking} onChange={e => update("smoking", e.target.value)} className="cursor-pointer">
                      <option value="">Select</option>
                      <option value="0">Non-Smoker</option>
                      <option value="1">Active Smoker</option>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <label className="text-xs font-semibold">Alcohol Intake</label>
                    <Select value={data.alcohol} onChange={e => update("alcohol", e.target.value)} className="cursor-pointer">
                      <option value="">Select</option>
                      <option value="0">None / Minimal</option>
                      <option value="1">Regular Consumption</option>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <label className="text-xs font-semibold">Physically Active</label>
                    <Select value={data.active} onChange={e => update("active", e.target.value)} className="cursor-pointer">
                      <option value="">Select</option>
                      <option value="1">Yes (Active)</option>
                      <option value="0">No (Sedentary)</option>
                    </Select>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Uploads */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card className="border-border shadow-sm" noPadding>
              <CardHeader className="px-6 py-4 border-b border-border bg-secondary/30 mb-0">
                <CardTitle className="text-sm font-bold text-muted-foreground flex items-center gap-2">
                  <FileText className="h-4 w-4 text-primary" /> ECG Upload
                </CardTitle>
              </CardHeader>
              <CardContent className="p-6 text-center">
                <div 
                  className={`border-2 border-dashed rounded-xl p-6 transition-all duration-300 ${dragActiveEcg ? 'border-primary bg-primary/5' : 'border-border bg-secondary/20 hover:bg-secondary/40'} ${files.ecg ? 'border-success/50 bg-success/5' : ''}`}
                  onDragEnter={e => handleDrag(e, setDragActiveEcg)}
                  onDragLeave={e => handleDrag(e, setDragActiveEcg)}
                  onDragOver={e => handleDrag(e, setDragActiveEcg)}
                  onDrop={e => handleDrop(e, "ecg", setDragActiveEcg)}
                >
                  <input type="file" ref={ecgInputRef} className="hidden" accept=".png,.jpg,.jpeg,.csv" onChange={e => setFile("ecg", e.target.files[0])} />
                  {!files.ecg ? (
                    <div className="flex flex-col items-center cursor-pointer" onClick={() => ecgInputRef.current?.click()}>
                      <UploadCloud className="h-8 w-8 text-muted-foreground mb-2" />
                      <p className="text-sm font-medium">Drag & Drop or Click to upload</p>
                      <p className="text-xs text-muted-foreground mt-1">PNG, JPG, CSV</p>
                    </div>
                  ) : (
                    <div className="flex flex-col items-center">
                      <CheckCircle2 className="h-8 w-8 text-success mb-2" />
                      <p className="text-sm font-medium text-success truncate w-full px-4">{files.ecg.name}</p>
                      <button type="button" onClick={() => setFile("ecg", null)} className="text-xs text-destructive flex items-center gap-1 mt-2 cursor-pointer hover:underline">
                        <Trash2 className="h-3 w-3" /> Remove File
                      </button>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>

            <Card className="border-border shadow-sm" noPadding>
              <CardHeader className="px-6 py-4 border-b border-border bg-secondary/30 mb-0">
                <CardTitle className="text-sm font-bold text-muted-foreground flex items-center gap-2">
                  <ScanHeart className="h-4 w-4 text-primary" /> Echo Upload
                </CardTitle>
              </CardHeader>
              <CardContent className="p-6 text-center">
                <div 
                  className={`border-2 border-dashed rounded-xl p-6 transition-all duration-300 ${dragActiveEcho ? 'border-primary bg-primary/5' : 'border-border bg-secondary/20 hover:bg-secondary/40'} ${files.echo ? 'border-success/50 bg-success/5' : ''}`}
                  onDragEnter={e => handleDrag(e, setDragActiveEcho)}
                  onDragLeave={e => handleDrag(e, setDragActiveEcho)}
                  onDragOver={e => handleDrag(e, setDragActiveEcho)}
                  onDrop={e => handleDrop(e, "echo", setDragActiveEcho)}
                >
                  <input type="file" ref={echoInputRef} className="hidden" accept=".mp4,.avi,.mov,.mkv,.png,.jpg" onChange={e => setFile("echo", e.target.files[0])} />
                  {!files.echo ? (
                    <div className="flex flex-col items-center cursor-pointer" onClick={() => echoInputRef.current?.click()}>
                      <UploadCloud className="h-8 w-8 text-muted-foreground mb-2" />
                      <p className="text-sm font-medium">Drag & Drop or Click to upload</p>
                      <p className="text-xs text-muted-foreground mt-1">MP4, AVI, PNG, JPG</p>
                    </div>
                  ) : (
                    <div className="flex flex-col items-center">
                      <CheckCircle2 className="h-8 w-8 text-success mb-2" />
                      <p className="text-sm font-medium text-success truncate w-full px-4">{files.echo.name}</p>
                      <button type="button" onClick={() => setFile("echo", null)} className="text-xs text-destructive flex items-center gap-1 mt-2 cursor-pointer hover:underline">
                        <Trash2 className="h-3 w-3" /> Remove File
                      </button>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Sticky Mobile Button, static on Desktop */}
          <div className="sticky bottom-4 z-20 md:static mt-8 shadow-xl md:shadow-none rounded-xl bg-background md:bg-transparent p-4 md:p-0 border md:border-none border-border">
            <Button
              disabled={!isReady || isProcessing}
              onClick={startAnalysis}
              className="w-full h-14 text-base font-bold shadow-lg hover:shadow-xl transition-all hover:-translate-y-0.5 cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isProcessing ? (
                <span className="flex items-center gap-2">
                  <BrainCircuit className="h-5 w-5 animate-pulse" /> Running Multi-Modal Analysis...
                </span>
              ) : (
                <span className="flex items-center gap-2">
                  <Zap className="h-5 w-5" /> Generate AI Prediction
                </span>
              )}
            </Button>
            {!isReady && (
              <p className="text-xs text-center text-muted-foreground mt-3 md:mt-2">
                * Please fill all required fields to enable prediction
              </p>
            )}
          </div>

        </div>

        {/* RIGHT COLUMN (35%) */}
        <div className="w-full lg:w-[35%] space-y-6">
          
          {/* Status Card */}
          <Card className="border-border shadow-sm overflow-hidden relative" noPadding>
            {isProcessing && (
              <motion.div 
                initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                className="absolute inset-0 bg-background/80 backdrop-blur-sm z-10 flex flex-col items-center justify-center p-6 text-center"
              >
                <div className="h-16 w-16 rounded-full bg-primary/20 flex items-center justify-center mb-4 border border-primary/30">
                  <ScanHeart className="h-8 w-8 text-primary animate-pulse" />
                </div>
                <h3 className="font-bold text-foreground">Processing Diagnostics</h3>
                <p className="text-xs text-muted-foreground mt-1 mb-4">Analyzing parameters through CardioAI neural net...</p>
                
                <div className="w-full h-2 rounded-full bg-secondary overflow-hidden">
                  <div className="h-full bg-primary transition-all duration-300 ease-out" style={{ width: `${progress}%` }} />
                </div>
                <span className="text-xs font-bold text-primary mt-2">{progress}%</span>
              </motion.div>
            )}

            <CardHeader className="px-6 py-4 bg-secondary/30 border-b border-border mb-0">
              <CardTitle className="text-sm font-bold uppercase tracking-wider text-muted-foreground flex items-center gap-2">
                <ShieldCheck className="h-4 w-4" /> Assessment Status
              </CardTitle>
            </CardHeader>
            <CardContent className="p-6 pb-8">
              <div className="flex flex-col items-center justify-center text-center space-y-3">
                <div className="h-24 w-24 rounded-full border-8 border-secondary flex items-center justify-center relative">
                  <span className="text-xs font-bold text-muted-foreground absolute bottom-2">N/A</span>
                  <div className="absolute inset-0 rounded-full border-8 border-transparent border-t-muted-foreground/20 rotate-45" />
                </div>
                <div>
                  <h3 className="text-lg font-bold text-foreground">Pending Data</h3>
                  <p className="text-sm text-muted-foreground mt-1">Submit the form to calculate your cardiovascular risk profile.</p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Quick Insights Placeholder */}
          <Card className="border-border shadow-sm bg-gradient-to-br from-primary/5 to-transparent" noPadding>
            <CardHeader className="px-6 py-4 mb-0 border-b border-border/50">
              <CardTitle className="text-sm font-bold flex items-center gap-2 text-foreground">
                <BrainCircuit className="h-4 w-4 text-primary" /> AI Quick Insights
              </CardTitle>
            </CardHeader>
            <CardContent className="p-6">
              <div className="space-y-3">
                <div className="h-4 w-3/4 bg-secondary rounded animate-pulse" />
                <div className="h-4 w-full bg-secondary rounded animate-pulse" />
                <div className="h-4 w-5/6 bg-secondary rounded animate-pulse" />
              </div>
              <p className="text-xs text-muted-foreground mt-4 italic text-center">
                Insights will appear here dynamically.
              </p>
            </CardContent>
          </Card>

          {/* Recent History Placeholder */}
          <Card className="border-border shadow-sm" noPadding>
            <CardHeader className="px-6 py-4 border-b border-border bg-secondary/30 mb-0">
              <CardTitle className="text-sm font-bold uppercase tracking-wider text-muted-foreground flex items-center gap-2">
                <Clock className="h-4 w-4" /> Recent History
              </CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              <div className="divide-y divide-border">
                {[1, 2, 3].map(i => (
                  <div key={i} className="p-4 flex items-center justify-between hover:bg-secondary/20 transition-colors cursor-pointer">
                    <div>
                      <div className="text-sm font-semibold text-foreground">Screening #{1004 - i}</div>
                      <div className="text-xs text-muted-foreground">Oct {14 - i}, 2023</div>
                    </div>
                    <ChevronRight className="h-4 w-4 text-muted-foreground" />
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

        </div>
      </div>
    </>
  );
}

// Simple internal icons to reduce lucide imports overhead
function UserIcon(props) {
  return <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
}
function DropletIcon(props) {
  return <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 22a7 7 0 0 0 7-7c0-2-1-3.9-3-5.5s-3.5-4-4-6.5c-.5 2.5-2 4.9-4 6.5C6 11.1 5 13 5 15a7 7 0 0 0 7 7z"/></svg>
}
function FlameIcon(props) {
  return <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z"/></svg>
}
