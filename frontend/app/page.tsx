"use client";

import { useState } from "react";
import { motion } from "motion/react";
import Image from "next/image";
import { Space_Grotesk } from "next/font/google";
import { streamF1Response, type Message } from "./actions";
export default function Home() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([
    { role: "model", text: "Welcome! \nState any query you have regarding the 2026 Formula 1 Regulations and I'll pull up snippets to help you understand!" }
  ]);
  const [isPending, setIsPending] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isPending) return;

    const userQuery = input;
    setInput("");
    setIsPending(true);

    setMessages((prev) => [
      ...prev,
      { role: "user", text: userQuery },
      { role: "model", text: "" } 
    ]);

    await streamF1Response(userQuery, (newChunk) => {
      setMessages((prev) => {
        const updated = [...prev];
        const lastIndex = updated.length - 1;
        updated[lastIndex] = {
          ...updated[lastIndex],
          text: updated[lastIndex].text + newChunk
        };
        return updated;
      });
    });

    setIsPending(false);
  };
  function formatMarkdownText(text: string) {
    const parts = text.split(/(\*\*.*?\*\*)/g);
    return parts.map((part, index) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return <strong key={index} className="font-bold text-black">{part.slice(2, -2)}</strong>;
      }
      return part;
    });
  }
  return (
    <>
      <div className=" flex rounded-xl *:rounded-xl bg-white shadow-xl shadow-black/20 *:mx-5 py-5 w-[99vw] h-[95vh] m-auto">
        <div className=" flex justify-center py-10 shadow-md shadow-black/50  items-center flex-col w-[25%] ">
          <div className="text-5xl font-bold  font-['Space_Grotesk']">RegExpert🏁</div>
          <p className="text-xl leading-10 italic">Formula 1 Regulations, simplified</p>
        </div>
        <div className="relative w-[75%] overflow-y-auto pb-32 outline outline-black/50 font-[Space_Grotesk] ">
          <div className="flex-1 overflow-y-auto space-y-4 pb-24 px-10 pt-10">
            {messages.map((msg, idx) => (
              <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <motion.div initial={{scale:0}} animate={{scale:1}}className={` max-w-[80%] overflow-auto rounded-2xl px-5 py-3 shadow-md ${
                  msg.role === 'user' ? 'bg-black text-white' : 'bg-gray-100 text-black'
                }`}>
                  <p className="text-xl whitespace-pre-wrap leading-relaxed">{formatMarkdownText(msg.text)}</p>
                </motion.div>
              </div>
            ))}
            {isPending && <p className="text-xl italic text-gray-400 animate-pulse">Analyzing F1 files...</p>}
          </div>
          <form onSubmit={handleSubmit}>
            <textarea 
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSubmit(e);
                }
              }}
              placeholder="Ask Away!" 
              disabled={isPending}
              className="w-[60%] h-[10%] backdrop-blur-md ml-[5vh] font-bold p-5 fixed bottom-[10%] shadow-xl shadow-black/10 rounded-2xl outline outline-black/50"
            ></textarea>
            <button 
              type="submit"
              disabled={isPending}
              className="rounded-[100%] bg-black w-20 h-20 fixed bottom-[10%] right-[3%] text-white font-bold text-3xl"
            >
              →
            </button>
          </form>
        </div>
      </div>
    </>
  );
}
