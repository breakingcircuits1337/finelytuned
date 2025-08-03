import React from "react";
// Update imports for UI primitives
import { Button } from "../components/ui/button";
import { Card } from "../components/ui/card";
import { Input } from "../components/ui/input";
import { Label } from "../components/ui/label";
import { Select } from "../components/ui/select";
import { Textarea } from "../components/ui/textarea";
import { Progress } from "../components/ui/progress";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "../components/ui/tabs";
import { Alert } from "../components/ui/alert";
import { ArrowRight } from "lucide-react";

export default function App() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-br from-gray-100 to-indigo-100">
      <Card className="max-w-lg w-full mx-auto shadow-lg p-8 mt-8">
        <h1 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <ArrowRight className="w-6 h-6 text-indigo-600" />
          AI Fine-Tuning App
        </h1>
        <form className="space-y-4">
          <Label htmlFor="prompt">Prompt</Label>
          <Textarea id="prompt" placeholder="Type your prompt..." />
          <Button type="submit" className="w-full">Submit</Button>
        </form>
        <Progress value={65} className="my-6" />
        <Tabs defaultValue="chat" className="">
          <TabsList>
            <TabsTrigger value="chat">Chat</TabsTrigger>
            <TabsTrigger value="train">Train</TabsTrigger>
          </TabsList>
          <TabsContent value="chat">
            <Alert>Chat with your fine-tuned model coming soon!</Alert>
          </TabsContent>
          <TabsContent value="train">
            <Alert>Training workflow UI coming soon!</Alert>
          </TabsContent>
        </Tabs>
      </Card>
    </div>
  );
}