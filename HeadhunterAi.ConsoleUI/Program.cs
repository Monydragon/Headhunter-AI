using LLama.Common;
using LLamaIntegration;

namespace HeadhunterAi.ConsoleUI;

public static class Program
    {
        public static async Task Main(string[] args)
        {
            var controller = new LLamaController();

            try
            {
                // Load the model from the specified path.
                // var existingModelPath = Path.Combine("M:\\", "phi-4-Q2_K.gguf");
                var existingModelPath = Path.Combine("M:\\", "phi-4-Q8_0.gguf");
                if(File.Exists(existingModelPath))
                {
                    // Initialize the controller with configurable parameters.
                    controller.Initialize(modelPath: existingModelPath);
                }
                else
                {
                    Console.WriteLine("The model file does not exist. Please provide the path to the model file:");
                    var modelPath = Console.ReadLine();
                    controller.Initialize(modelPath: modelPath);
                }

                // Set up a chat session with optional chat history configuration.
                controller.SetupChatSession(chatHistory =>
                {
                    chatHistory.AddMessage(AuthorRole.System, "Transcript of a dialog, where the User interacts with an Assistant named Bob. Bob is helpful, kind, honest, good at writing, and never fails to answer the User's requests immediately and with precision.");
                    chatHistory.AddMessage(AuthorRole.User, "Hello, Bob.");
                    chatHistory.AddMessage(AuthorRole.Assistant, "Hello. How may I help you today?");
                });

                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine("The chat session has started. Type 'exit' to quit.");
                Console.ResetColor();

                string userInput;
                do
                {
                    Console.ForegroundColor = ConsoleColor.Green;
                    Console.Write("User: ");
                    Console.ResetColor();
                    userInput = Console.ReadLine() ?? string.Empty;

                    if (userInput.ToLower() == "exit")
                    {
                        break;
                    }

                    await foreach (var response in controller.ProcessUserInputAsync(userInput))
                    {
                        Console.ForegroundColor = ConsoleColor.White;
                        Console.Write(response);
                    }
                    Console.WriteLine();
                } while (true);
            }
            finally
            {
                controller.Dispose();
            }
        }
    }