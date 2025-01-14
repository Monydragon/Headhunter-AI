using System;
using System.IO;
using System.Threading.Tasks;
using HeadhunterAI.Core;
using LLama.Common;

namespace HeadhunterAI.ConsoleUI
{
    /// <summary>
    /// Entry point for the HeadhunterAI application. Sets up and interacts with the LLamaController.
    /// </summary>
    public static class Program
    {
        public static async Task Main(string[] args)
        {
            // Create an instance of LLamaController
            using var controller = new LLamaController();

            try
            {
                // Load or request the model path.
                string modelPath = GetModelPath();

                // Initialize the controller with optimized parameters.
                controller.Initialize(
                    modelPath: modelPath,
                    contextSize: 1024,  // Adjust based on your memory and conversation length needs.
                    gpuLayerCount: 20   // Increase if your GPU supports more layers for better performance.
                );

                // Set up the chat session with pre-defined history.
                controller.SetupChatSession(chatHistory =>
                {
                    chatHistory.AddMessage(AuthorRole.System,
                        "Transcript of a dialog, where the User interacts with an Assistant named Bob. Bob is helpful, kind, honest, good at writing, and never fails to answer the User's requests immediately and with precision.");
                    chatHistory.AddMessage(AuthorRole.User, "Hello, Bob.");
                    chatHistory.AddMessage(AuthorRole.Assistant, "Hello. How may I help you today?");
                });

                // Notify the user that the session has started.
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine("The chat session has started. Type 'exit' to quit.");
                Console.ResetColor();

                // Begin the chat interaction loop.
                await RunChatSessionAsync(controller);
            }
            catch (Exception ex)
            {
                // Handle exceptions and notify the user.
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"An error occurred: {ex.Message}");
                Console.ResetColor();
            }
        }

        /// <summary>
        /// Gets the model path from either a predefined location or user input.
        /// </summary>
        /// <returns>The valid path to the model file.</returns>
        private static string GetModelPath()
        {
            // Default model path
            string predefinedPath = Path.Combine("M:\\", "phi-4-Q2_K.gguf");

            // Check if the predefined model path exists.
            if (File.Exists(predefinedPath))
            {
                return predefinedPath;
            }

            // Request the user to input the model path if the predefined one doesn't exist.
            Console.WriteLine("The model file does not exist at the predefined location. Please provide the path:");
            string? userInputPath = Console.ReadLine();
            if (string.IsNullOrEmpty(userInputPath) || !File.Exists(userInputPath))
            {
                throw new FileNotFoundException("The specified model file was not found.");
            }

            return userInputPath;
        }

        /// <summary>
        /// Handles the chat session interaction loop with streaming token output.
        /// </summary>
        /// <param name="controller">The LLamaController handling the AI model.</param>
        private static async Task RunChatSessionAsync(LLamaController controller)
        {
            string userInput;
            do
            {
                // Prompt the user for input.
                Console.ForegroundColor = ConsoleColor.Green;
                Console.Write("User: ");
                Console.ResetColor();
                userInput = Console.ReadLine() ?? string.Empty;

                // Exit the loop if the user types "exit".
                if (userInput.ToLower() == "exit") break;

                Console.ForegroundColor = ConsoleColor.White;

                try
                {
                    // Stream tokens as they are generated.
                    await foreach (var token in controller.ProcessUserInputAsync(userInput, maxTokens: 256))
                    {
                        Console.Write(token);
                    }
                    Console.WriteLine(); // Add a newline after the response.
                }
                catch (Exception ex)
                {
                    // Handle errors during response generation.
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine($"Error generating response: {ex.Message}");
                    Console.ResetColor();
                }

                Console.ResetColor();
            } while (true);
        }
    }
}
