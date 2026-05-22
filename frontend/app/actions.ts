
export interface Message {
  role: "user" | "model";
  text: string;
}

export async function streamF1Response(
  userPrompt: string,
  onTokenReceived: (token: string) => void
): Promise<string> {
  try {
    const response = await fetch("http://localhost:8000/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ message: userPrompt }),
    });

    if (!response.ok) {
      throw new Error(`FastAPI responded with status: ${response.status}`);
    }

    const reader = response.body?.getReader();
    const decoder = new TextDecoder("utf-8");
    
    if (!reader) {
      throw new Error("Failed to initialize text stream reader from API response.");
    }

    let fullText = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const textChunk = decoder.decode(value, { stream: true });
      fullText += textChunk;

      onTokenReceived(textChunk);
    }

    return fullText;

  } catch (error) {
    console.error("Failed to connect to FastAPI microservice:", error);
    const errorMessage = "⚠️ System Error: Unable to communicate with backend.";
    onTokenReceived(errorMessage);
    return errorMessage;
  }
}