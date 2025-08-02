import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button.jsx'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Input } from '@/components/ui/input.jsx'
import { Label } from '@/components/ui/label.jsx'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select.jsx'
import { Textarea } from '@/components/ui/textarea.jsx'
import { Progress } from '@/components/ui/progress.jsx'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs.jsx'
import { Alert, AlertDescription } from '@/components/ui/alert.jsx'
import { Upload, Play, Square, Cpu, Zap, Clock, FileText } from 'lucide-react'
import './App.css'

function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [uploadedFile, setUploadedFile] = useState(null)
  const [models, setModels] = useState([])
  const [selectedModel, setSelectedModel] = useState('')
  const [device, setDevice] = useState('cpu')
  const [epochs, setEpochs] = useState(3)
  const [learningRate, setLearningRate] = useState(5e-5)
  const [batchSize, setBatchSize] = useState(4)
  const [systemPrompt, setSystemPrompt] = useState('')
  const [pastedText, setPastedText] = useState('')
  const [trainingMode, setTrainingMode] = useState('file')
  const [estimation, setEstimation] = useState(null)
  const [progress, setProgress] = useState(null)
  const [isTraining, setIsTraining] = useState(false)
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')

  // Fetch available models on component mount
  useEffect(() => {
    fetchModels()
  }, [])

  // Poll training progress when training is active
  useEffect(() => {
    let interval
    if (isTraining) {
      interval = setInterval(fetchProgress, 2000)
    }
    return () => {
      if (interval) clearInterval(interval)
    }
  }, [isTraining])

  const fetchModels = async () => {
    try {
      const response = await fetch('/api/training/models')
      const data = await response.json()
      setModels(data)
      if (data.length > 0) {
        setSelectedModel(data[0].name)
      }
    } catch (err) {
      setError('Failed to fetch models')
    }
  }

  const handleFileSelect = (event) => {
    const file = event.target.files[0]
    setSelectedFile(file)
    setUploadedFile(null)
    setEstimation(null)
  }

  const uploadFile = async () => {
    if (!selectedFile) return

    const formData = new FormData()
    formData.append('file', selectedFile)

    try {
      setError('')
      const response = await fetch('/api/training/upload', {
        method: 'POST',
        body: formData
      })
      
      const data = await response.json()
      if (response.ok) {
        setUploadedFile(data)
        setSuccess('File uploaded successfully!')
        setTimeout(() => setSuccess(''), 3000)
      } else {
        setError(data.error || 'Upload failed')
      }
    } catch (err) {
      setError('Upload failed: ' + err.message)
    }
  }

  const estimateTrainingTime = async () => {
    if (!uploadedFile && trainingMode === 'file') return

    try {
      setError('')
      const response = await fetch('/api/training/estimate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          file_path: uploadedFile?.file_path,
          device,
          model_name: selectedModel
        })
      })
      
      const data = await response.json()
      if (response.ok) {
        setEstimation(data)
      } else {
        setError(data.error || 'Estimation failed')
      }
    } catch (err) {
      setError('Estimation failed: ' + err.message)
    }
  }

  const startTraining = async () => {
    try {
      setError('')
      const config = {
        device,
        model_name: selectedModel,
        epochs,
        learning_rate: learningRate,
        batch_size: batchSize
      }

      if (trainingMode === 'file' && uploadedFile) {
        config.file_path = uploadedFile.file_path
      } else if (trainingMode === 'prompt') {
        config.system_prompt = systemPrompt
      } else if (trainingMode === 'text') {
        config.pasted_text = pastedText
      }

      const response = await fetch('/api/training/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      })
      
      const data = await response.json()
      if (response.ok) {
        setIsTraining(true)
        setSuccess('Training started!')
        setTimeout(() => setSuccess(''), 3000)
      } else {
        setError(data.error || 'Failed to start training')
      }
    } catch (err) {
      setError('Failed to start training: ' + err.message)
    }
  }

  const stopTraining = async () => {
    try {
      const response = await fetch('/api/training/stop', {
        method: 'POST'
      })
      
      if (response.ok) {
        setIsTraining(false)
        setSuccess('Training stopped')
        setTimeout(() => setSuccess(''), 3000)
      }
    } catch (err) {
      setError('Failed to stop training: ' + err.message)
    }
  }

  const fetchProgress = async () => {
    try {
      const response = await fetch('/api/training/progress')
      const data = await response.json()
      setProgress(data)
      
      if (data.status === 'completed' || data.status === 'error' || data.status === 'stopped') {
        setIsTraining(false)
      }
    } catch (err) {
      console.error('Failed to fetch progress:', err)
    }
  }

  const formatTime = (seconds) => {
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    return `${hours}h ${minutes}m`
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 to-blue-50 p-4">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            🐌 Slow Go Mofo
          </h1>
          <p className="text-lg text-gray-600">
            AI Fine-tuning Made Simple (and Slow)
          </p>
        </div>

        {error && (
          <Alert className="mb-4 border-red-200 bg-red-50">
            <AlertDescription className="text-red-800">{error}</AlertDescription>
          </Alert>
        )}

        {success && (
          <Alert className="mb-4 border-green-200 bg-green-50">
            <AlertDescription className="text-green-800">{success}</AlertDescription>
          </Alert>
        )}

        <div className="grid gap-6">
          {/* Training Data Input */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileText className="h-5 w-5" />
                Training Data
              </CardTitle>
              <CardDescription>
                Choose your training data source
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Tabs value={trainingMode} onValueChange={setTrainingMode}>
                <TabsList className="grid w-full grid-cols-3">
                  <TabsTrigger value="file">Upload File</TabsTrigger>
                  <TabsTrigger value="prompt">System Prompt</TabsTrigger>
                  <TabsTrigger value="text">Paste Text</TabsTrigger>
                </TabsList>
                
                <TabsContent value="file" className="space-y-4">
                  <div>
                    <Label htmlFor="file">Dataset File (.txt, .csv, .jsonl)</Label>
                    <Input
                      id="file"
                      type="file"
                      accept=".txt,.csv,.jsonl,.json"
                      onChange={handleFileSelect}
                      className="mt-1"
                    />
                  </div>
                  {selectedFile && (
                    <Button onClick={uploadFile} className="w-full">
                      <Upload className="h-4 w-4 mr-2" />
                      Upload File
                    </Button>
                  )}
                  {uploadedFile && (
                    <div className="p-3 bg-green-50 rounded-lg">
                      <p className="text-sm text-green-800">
                        ✅ {uploadedFile.filename} ({(uploadedFile.file_size / 1024 / 1024).toFixed(2)} MB)
                      </p>
                    </div>
                  )}
                </TabsContent>
                
                <TabsContent value="prompt" className="space-y-4">
                  <div>
                    <Label htmlFor="system-prompt">System Prompt</Label>
                    <Textarea
                      id="system-prompt"
                      placeholder="Enter your system prompt for fine-tuning..."
                      value={systemPrompt}
                      onChange={(e) => setSystemPrompt(e.target.value)}
                      className="mt-1 min-h-[100px]"
                    />
                  </div>
                </TabsContent>
                
                <TabsContent value="text" className="space-y-4">
                  <div>
                    <Label htmlFor="pasted-text">Training Text</Label>
                    <Textarea
                      id="pasted-text"
                      placeholder="Paste your training text here..."
                      value={pastedText}
                      onChange={(e) => setPastedText(e.target.value)}
                      className="mt-1 min-h-[100px]"
                    />
                  </div>
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>

          {/* Model Configuration */}
          <Card>
            <CardHeader>
              <CardTitle>Model Configuration</CardTitle>
              <CardDescription>
                Configure your training parameters
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="model">Model</Label>
                  <Select value={selectedModel} onValueChange={setSelectedModel}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select a model" />
                    </SelectTrigger>
                    <SelectContent>
                      {models.map((model) => (
                        <SelectItem key={model.name} value={model.name}>
                          {model.description}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                
                <div>
                  <Label htmlFor="device">Device</Label>
                  <Select value={device} onValueChange={setDevice}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="cpu">
                        <div className="flex items-center gap-2">
                          <Cpu className="h-4 w-4" />
                          CPU (Slow but Steady)
                        </div>
                      </SelectItem>
                      <SelectItem value="gpu">
                        <div className="flex items-center gap-2">
                          <Zap className="h-4 w-4" />
                          GPU (Fast if Available)
                        </div>
                      </SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <Label htmlFor="epochs">Epochs</Label>
                  <Input
                    id="epochs"
                    type="number"
                    min="1"
                    max="10"
                    value={epochs}
                    onChange={(e) => setEpochs(parseInt(e.target.value))}
                  />
                </div>
                
                <div>
                  <Label htmlFor="learning-rate">Learning Rate</Label>
                  <Input
                    id="learning-rate"
                    type="number"
                    step="0.00001"
                    value={learningRate}
                    onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                  />
                </div>
                
                <div>
                  <Label htmlFor="batch-size">Batch Size</Label>
                  <Input
                    id="batch-size"
                    type="number"
                    min="1"
                    max="16"
                    value={batchSize}
                    onChange={(e) => setBatchSize(parseInt(e.target.value))}
                  />
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Time Estimation */}
          {(uploadedFile || systemPrompt || pastedText) && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Clock className="h-5 w-5" />
                  Training Time Estimation
                </CardTitle>
              </CardHeader>
              <CardContent>
                <Button onClick={estimateTrainingTime} className="mb-4">
                  Estimate Training Time
                </Button>
                
                {estimation && (
                  <div className="p-4 bg-blue-50 rounded-lg">
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                      <div>
                        <p className="font-semibold">Samples</p>
                        <p>{estimation.sample_count}</p>
                      </div>
                      <div>
                        <p className="font-semibold">File Size</p>
                        <p>{estimation.file_size_mb} MB</p>
                      </div>
                      <div>
                        <p className="font-semibold">Device</p>
                        <p>{estimation.device.toUpperCase()}</p>
                      </div>
                      <div>
                        <p className="font-semibold">Estimated Time</p>
                        <p>{estimation.estimated_hours}h {estimation.estimated_minutes}m</p>
                      </div>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          {/* Training Controls */}
          <Card>
            <CardHeader>
              <CardTitle>Training Control</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex gap-4">
                <Button
                  onClick={startTraining}
                  disabled={isTraining || (!uploadedFile && !systemPrompt && !pastedText)}
                  className="flex-1"
                >
                  <Play className="h-4 w-4 mr-2" />
                  Start Training
                </Button>
                
                <Button
                  onClick={stopTraining}
                  disabled={!isTraining}
                  variant="destructive"
                >
                  <Square className="h-4 w-4 mr-2" />
                  Stop
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Training Progress */}
          {progress && (
            <Card>
              <CardHeader>
                <CardTitle>Training Progress</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <span>Progress</span>
                    <span>{Math.round(progress.progress)}%</span>
                  </div>
                  <Progress value={progress.progress} className="w-full" />
                </div>
                
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <p className="font-semibold">Status</p>
                    <p className="capitalize">{progress.status}</p>
                  </div>
                  <div>
                    <p className="font-semibold">Epoch</p>
                    <p>{progress.current_epoch}/{progress.total_epochs}</p>
                  </div>
                  <div>
                    <p className="font-semibold">Loss</p>
                    <p>{progress.loss?.toFixed(4) || 'N/A'}</p>
                  </div>
                  <div>
                    <p className="font-semibold">Time Remaining</p>
                    <p>{progress.estimated_time_remaining ? formatTime(progress.estimated_time_remaining) : 'N/A'}</p>
                  </div>
                </div>
                
                <div className="p-3 bg-gray-50 rounded-lg">
                  <p className="text-sm text-gray-700">{progress.message}</p>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}

export default App

