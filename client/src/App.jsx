import React, { useRef, useState } from "react";
import "./App.css";
import { Button } from "./components/ui/button";
import { Card } from "./components/ui/card";
import { Input } from "./components/ui/input";
import { Label } from "./components/ui/label";
import { Select } from "./components/ui/select";
import { Textarea } from "./components/ui/textarea";
import { Progress } from "./components/ui/progress";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "./components/ui/tabs";
import { Alert } from "./components/ui/alert";
import { Toggle } from "./components/ui/toggle";
import { Download, Send, UploadCloud, CloudUpload } from "lucide-react";

const MODEL_OPTIONS = [
  { value: "gpt2", label: "GPT-2 Small" },
  { value: "gpt2-medium", label: "GPT-2 Medium" },
  { value: "distilgpt2", label: "DistilGPT-2" },
  { value: "microsoft/DialoGPT-small", label: "DialoGPT Small" }
];

export default function App() {
  // Training form state
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState("");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [pastedText, setPastedText] = useState("");
  const [model, setModel] = useState(MODEL_OPTIONS[0].value);
  const [device, setDevice] = useState("cpu");
  const [epochs, setEpochs] = useState(3);
  const [batchSize, setBatchSize] = useState(4);
  const [learningRate, setLearningRate] = useState(5e-5);
  const [estimation, setEstimation] = useState(null);
  const [estimating, setEstimating] = useState(false);

  // Training progress
  const [progress, setProgress] = useState({
    status: "idle",
    progress: 0,
    message: "",
    current_epoch: 0,
    total_epochs: 0,
    loss: 0,
    estimated_time_remaining: 0,
  });
  const [trainingStarted, setTrainingStarted] = useState(false);
  const [polling, setPolling] = useState(false);

  // Chat section
  const [chatInput, setChatInput] = useState("");
  const [chatFeed, setChatFeed] = useState([]);
  const [chatting, setChatting] = useState(false);

  // Download model
  const [downloading, setDownloading] = useState(false);

  // HF push
  const [repoName, setRepoName] = useState("");
  const [repoPrivate, setRepoPrivate] = useState(true);
  const [pushStatus, setPushStatus] = useState("");
  const [pushing, setPushing] = useState(false);

  // UI state
  const [tab, setTab] = useState("train");
  const fileInputRef = useRef();

  // Utility
  const canStartTraining =
    (file || systemPrompt.trim() || pastedText.trim()) && !polling && (progress.status === "idle" || progress.status === "completed" || progress.status === "error" || progress.status === "stopped");

  // Estimate training time
  async function handleEstimate(e) {
    e.preventDefault();
    setEstimating(true);
    setEstimation(null);
    let filePath = null;
    if (file) {
      // Need to upload first to get file_path
      filePath = await uploadTempFile();
      if (!filePath) {
        setEstimating(false);
        return;
      }
    } else if (progress.file_path) {
      filePath = progress.file_path;
    }
    try {
      const resp = await fetch("/api/training/estimate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          file_path: filePath,
          device,
          model_name: model,
        }),
      });
      const result = await resp.json();
      setEstimation(result);
    } catch (err) {
      setEstimation({ error: "Failed to estimate." });
    }
    setEstimating(false);
  }

  // Upload file to backend
  async function uploadTempFile() {
    if (!file) return null;
    const form = new FormData();
    form.append("file", file);
    try {
      const resp = await fetch("/api/training/upload", {
        method: "POST",
        body: form,
      });
      const result = await resp.json();
      if (result.file_path) {
        setProgress((p) => ({ ...p, file_path: result.file_path }));
        return result.file_path;
      } else {
        alert("Upload failed: " + (result.error || "Unknown error"));
        return null;
      }
    } catch (err) {
      alert("Upload error");
      return null;
    }
  }

  // Start training
  async function handleStartTraining(e) {
    e.preventDefault();
    setTrainingStarted(true);
    setPushStatus("");
    let filePath = null;
    if (file) {
      filePath = await uploadTempFile();
      if (!filePath) {
        setTrainingStarted(false);
        return;
      }
    } else if (progress.file_path) {
      filePath = progress.file_path;
    }
    const body = {
      file_path: filePath,
      system_prompt: systemPrompt,
      pasted_text: pastedText,
      model_name: model,
      device,
      epochs: Number(epochs),
      batch_size: Number(batchSize),
      learning_rate: Number(learningRate)
    };
    try {
      const resp = await fetch("/api/training/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const result = await resp.json();
      if (resp.ok) {
        setPolling(true);
        pollProgress();
      } else {
        alert(result.error || "Failed to start training");
        setTrainingStarted(false);
      }
    } catch (err) {
      alert("Failed to start training");
      setTrainingStarted(false);
    }
  }

  // Poll progress
  async function pollProgress() {
    setPolling(true);
    let done = false;
    while (!done) {
      await new Promise((res) => setTimeout(res, 1500));
      try {
        const resp = await fetch("/api/training/progress");
        const data = await resp.json();
        setProgress(data);
        done =
          ["completed", "error", "stopped"].includes(data.status) ||
          !polling;
        if (done) {
          setPolling(false);
          setTrainingStarted(false);
        }
      } catch {
        setPolling(false);
        setTrainingStarted(false);
        break;
      }
    }
  }

  // Stop training
  async function handleStopTraining() {
    await fetch("/api/training/stop", { method: "POST" });
    setPolling(false);
    setTrainingStarted(false);
  }

  // Chat with model
  async function handleSendChat(e) {
    e.preventDefault();
    if (!chatInput.trim()) return;
    const prompt = chatInput;
    setChatFeed((feed) => [...feed, { role: "user", content: prompt }]);
    setChatting(true);
    setChatInput("");
    try {
      const resp = await fetch("/api/training/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
      });
      const data = await resp.json();
      setChatFeed((feed) => [
        ...feed,
        { role: "assistant", content: data.reply || data.error || "No reply." }
      ]);
    } catch (err) {
      setChatFeed((feed) => [
        ...feed,
        { role: "assistant", content: "Failed to get reply." }
      ]);
    }
    setChatting(false);
  }

  // Download model
  async function handleDownloadModel() {
    setDownloading(true);
    try {
      const resp = await fetch("/api/training/download");
      if (!resp.ok) {
        alert("Failed to download model.");
        setDownloading(false);
        return;
      }
      const blob = await resp.blob();
      // Trigger download
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "fine_tuned_model.zip";
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch {
      alert("Download failed");
    }
    setDownloading(false);
  }

  // Push to Hugging Face
  async function handlePushHF(e) {
    e.preventDefault();
    setPushing(true);
    setPushStatus("");
    try {
      const resp = await fetch("/api/training/push_hf", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          repo_name: repoName,
          private: repoPrivate
        }),
      });
      const data = await resp.json();
      if (data.success) {
        setPushStatus("Model pushed! View at: " + data.repo_url);
      } else {
        setPushStatus(data.error || "Push failed.");
      }
    } catch {
      setPushStatus("Push failed.");
    }
    setPushing(false);
  }

  // Handle file input change
  function handleFileChange(e) {
    const f = e.target.files[0];
    setFile(f);
    setFileName(f ? f.name : "");
  }

  // Handle tab change
  function handleTabChange(value) {
    setTab(value);
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-100 to-indigo-100 flex flex-col items-center py-10">
      <Card className="w-full max-w-2xl mx-auto shadow-lg p-6">
        <h1 className="text-2xl font-bold mb-2 text-indigo-700 flex items-center gap-2">
          <UploadCloud className="w-6 h-6" />
          AI Fine-Tuning App
        </h1>

        <Tabs defaultValue={tab} className="mt-4" value={tab}>
          <TabsList>
            <TabsTrigger value="train" onClick={() => handleTabChange("train")}>Train</TabsTrigger>
            <TabsTrigger value="chat" onClick={() => handleTabChange("chat")} disabled={progress.status !== "completed"}>Chat</TabsTrigger>
            <TabsTrigger value="model" onClick={() => handleTabChange("model")} disabled={progress.status !== "completed"}>Model</TabsTrigger>
          </TabsList>
          <TabsContent value="train">
            <form className="space-y-5" onSubmit={handleStartTraining}>
              <div>
                <Label>Dataset file</Label>
                <div className="flex items-center gap-2">
                  <Input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleFileChange}
                    accept=".txt,.csv,.jsonl,.json"
                  />
                  {fileName && (
                    <span className="text-xs text-gray-600">{fileName}</span>
                  )}
                </div>
                <div className="flex gap-2 mt-2">
                  <Button type="button" onClick={() => { setFile(null); setFileName(""); if (fileInputRef.current) fileInputRef.current.value = ""; }} variant="outline" className="bg-gray-200 text-gray-700">Clear</Button>
                </div>
              </div>
              <div>
                <Label>System prompt (optional)</Label>
                <Textarea
                  rows={2}
                  value={systemPrompt}
                  onChange={e => setSystemPrompt(e.target.value)}
                  placeholder="e.g. You are a helpful assistant."
                />
              </div>
              <div>
                <Label>Pasted text (optional)</Label>
                <Textarea
                  rows={2}
                  value={pastedText}
                  onChange={e => setPastedText(e.target.value)}
                  placeholder="Paste raw training text here"
                />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label>Model</Label>
                  <Select value={model} onChange={e => setModel(e.target.value)}>
                    {MODEL_OPTIONS.map(opt => (
                      <option value={opt.value} key={opt.value}>{opt.label}</option>
                    ))}
                  </Select>
                </div>
                <div>
                  <Label>Device</Label>
                  <Select value={device} onChange={e => setDevice(e.target.value)}>
                    <option value="cpu">CPU</option>
                    <option value="gpu">GPU</option>
                  </Select>
                </div>
              </div>
              <div className="grid grid-cols-3 gap-4">
                <div>
                  <Label>Epochs</Label>
                  <Input type="number" min={1} max={20} value={epochs} onChange={e => setEpochs(e.target.value)} />
                </div>
                <div>
                  <Label>Batch size</Label>
                  <Input type="number" min={1} max={64} value={batchSize} onChange={e => setBatchSize(e.target.value)} />
                </div>
                <div>
                  <Label>Learning rate</Label>
                  <Input type="number" step="any" value={learningRate} onChange={e => setLearningRate(e.target.value)} />
                </div>
              </div>
              <div className="flex gap-3 items-center">
                <Button type="button" onClick={handleEstimate} disabled={estimating}>
                  Estimate time
                </Button>
                {estimation && (
                  <span className="text-sm text-gray-700">
                    {estimation.error
                      ? estimation.error
                      : `~${estimation.estimated_minutes} min (${estimation.sample_count} samples)`}
                  </span>
                )}
              </div>
              <div className="flex gap-3 items-center">
                <Button type="submit" disabled={!canStartTraining || trainingStarted}>
                  Start Training
                </Button>
                <Button type="button" onClick={handleStopTraining} disabled={!polling} variant="outline" className="bg-gray-300 text-gray-700">
                  Stop
                </Button>
                <span className="text-sm text-gray-700">{progress.message}</span>
              </div>
              <Progress value={progress.progress || 0} />
              <div className="flex gap-4">
                <span className="text-xs text-gray-600">Status: {progress.status}</span>
                {progress.current_epoch > 0 && (
                  <span className="text-xs text-gray-600">
                    Epoch: {progress.current_epoch}/{progress.total_epochs}
                  </span>
                )}
                {progress.loss > 0 && (
                  <span className="text-xs text-gray-600">Loss: {progress.loss.toFixed(3)}</span>
                )}
              </div>
              {progress.status === "completed" && (
                <Alert className="mt-4 bg-green-50 border-green-200 text-green-800">
                  Training completed successfully! You can now chat with your model, download it, or push to Hugging Face.
                </Alert>
              )}
              {progress.status === "error" && (
                <Alert className="mt-4 bg-red-50 border-red-200 text-red-700">
                  Error: {progress.message}
                </Alert>
              )}
            </form>
          </TabsContent>
          <TabsContent value="chat">
            {progress.status !== "completed" ? (
              <Alert>You must finish training before chatting with your model.</Alert>
            ) : (
              <div className="flex flex-col space-y-4">
                <div className="h-64 overflow-y-auto rounded bg-gray-100 p-2 border border-gray-200 mb-2">
                  {chatFeed.length === 0 && (
                    <div className="text-gray-400 text-center pt-10">No messages yet.</div>
                  )}
                  {chatFeed.map((msg, i) => (
                    <div
                      key={i}
                      className={
                        msg.role === "user"
                          ? "text-right mb-2"
                          : "text-left mb-2"
                      }
                    >
                      <span
                        className={
                          msg.role === "user"
                            ? "inline-block bg-indigo-100 text-indigo-800 px-3 py-1 rounded-lg"
                            : "inline-block bg-white border text-gray-700 px-3 py-1 rounded-lg"
                        }
                      >
                        {msg.content}
                      </span>
                    </div>
                  ))}
                </div>
                <form className="flex gap-2" onSubmit={handleSendChat}>
                  <Input
                    value={chatInput}
                    onChange={e => setChatInput(e.target.value)}
                    placeholder="Type message..."
                    className="flex-1"
                    disabled={chatting}
                  />
                  <Button type="submit" disabled={chatting || !chatInput.trim()}>
                    <Send className="w-4 h-4 inline mr-1" />
                    Send
                  </Button>
                </form>
              </div>
            )}
          </TabsContent>
          <TabsContent value="model">
            <div className="flex flex-col gap-4">
              <div className="flex gap-2">
                <Button
                  onClick={handleDownloadModel}
                  disabled={progress.status !== "completed" || downloading}
                >
                  <Download className="w-4 h-4 inline mr-1" />
                  Download Model
                </Button>
                {downloading && <span className="text-sm text-gray-500">Downloading...</span>}
              </div>
              <form className="flex flex-col gap-2 max-w-sm" onSubmit={handlePushHF}>
                <Label>Push to Hugging Face Hub</Label>
                <Input
                  type="text"
                  value={repoName}
                  onChange={e => setRepoName(e.target.value)}
                  placeholder="Repository name (e.g. my-finetune-bot)"
                  required
                />
                <label className="flex items-center gap-2 text-sm">
                  <Toggle
                    checked={repoPrivate}
                    onChange={e => setRepoPrivate(e.target.checked)}
                  />
                  Private repository
                </label>
                <Button
                  type="submit"
                  disabled={!repoName || pushing || progress.status !== "completed"}
                >
                  <CloudUpload className="w-4 h-4 inline mr-1" />
                  Push to HF
                </Button>
                {pushStatus && (
                  <Alert className="mt-2">{pushStatus}</Alert>
                )}
              </form>
            </div>
          </TabsContent>
        </Tabs>
      </Card>
    </div>
  );
}