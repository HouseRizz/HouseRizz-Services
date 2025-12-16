# Virtual Staging API

This service provides AI-powered virtual staging capabilities, allowing users to redesign rooms with selected furniture.

## Base URL
Deployed URL: `https://virtual-staging-api-wmggns3mvq-uc.a.run.app`

## Endpoints

### 1. System
- **GET /health**: Check service status.
  - Response: `{"status": "healthy", "version": "0.1.0"}`

### 2. Inventory Management
- **POST /ingest**: Trigger ingestion of furniture images from the local `furniture_images` directory into the vector database.
  - Query Param: `reset=true` (optional) to clear existing data.
- **GET /inventory**: Get total count of indexed furniture items.
- **GET /furniture/{filepath}**: Serve a furniture image file.

### 3. Discovery & Selection
- **GET /search**: Semantic search for furniture.
  - Query Params: `query` (e.g. "modern sofa"), `top_k` (default 5).
- **POST /select/upload**: Select furniture based on a room image analysis.
  - Body:
    ```json
    {
      "room_image_base64": "...",
      "mime_type": "image/jpeg",
      "vibe_text": "Scandinavian modern",
      "top_k": 5
    }
    ```
  - Returns: List of recommended furniture items from inventory.

### 4. Virtual Staging (Generation)

#### Auto-Design (Full Pipeline)
- **POST /design/upload**: Automatically analyzes the room, selects furniture, and generates a redesign.
  - Body:
    ```json
    {
      "room_image_base64": "...",
      "mime_type": "image/jpeg",
      "vibe_text": "Modern Minimalist"
    }
    ```

#### Manual Design (Custom Furniture)
- **POST /generate**: Generate a staged room using specific furniture items (manual selection).
  - Body:
    ```json
    {
      "room_image_base64": "...",
      "room_mime_type": "image/jpeg",
      "vibe_text": "Modern Minimalist",
      "furniture_items": [
        {
          "name": "Emerald Sofa",
          "image_base64": "...",
          "mime_type": "image/jpeg"
        }
      ]
    }
    ```

## Frontend Implementation Guide

### Uploading Images & Generating
When building a frontend (React, Vue, Next.js, etc.), you will typically handle file inputs and convert them to base64 before sending to the API.

#### 1. Helper: Convert File to Base64
```javascript
const fileToBase64 = (file) => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => {
        // Remove "data:image/jpeg;base64," prefix
        const base64String = reader.result.split(',')[1]; 
        resolve(base64String);
    };
    reader.onerror = (error) => reject(error);
  });
};
```

#### 2. Example: Manual Staging Flow
```javascript
async function generateStagedRoom(roomFile, furnitureFiles, vibe) {
  const roomBase64 = await fileToBase64(roomFile);
  
  const furnitureItems = await Promise.all(furnitureFiles.map(async (file) => ({
    name: file.name,
    image_base64: await fileToBase64(file),
    mime_type: file.type
  })));

  const payload = {
    room_image_base64: roomBase64,
    room_mime_type: roomFile.type,
    vibe_text: vibe,
    furniture_items: furnitureItems
  };

  try {
    const response = await fetch('https://virtual-staging-api-wmggns3mvq-uc.a.run.app/generate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload)
    });

    if (!response.ok) throw new Error('Generation failed');

    const data = await response.json();
    console.log("Generated Image URL:", data.generated_image_url);
    return data.generated_image_url;
    
  } catch (error) {
    console.error("Error generating room:", error);
  }
}
```

### Displaying Generated Images
The `generated_image_url` returned by the API is a public Google Cloud Storage URL. You can display it directly in an `<img>` tag.

```jsx
<img src={generatedImageUrl} alt="Staged Room" />
```
