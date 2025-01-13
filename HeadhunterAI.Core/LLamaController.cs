using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using LLama;
using LLama.Common;

namespace LLamaIntegration
{
    /// <summary>
    /// Handles the initialization and execution of the LLama model with modular and configurable options.
    /// </summary>
    public class LLamaController : IDisposable
    {
        private LLamaWeights? _model;
        private LLamaContext? _context;
        private InteractiveExecutor? _executor;
        private ChatSession? _session;

        /// <summary>
        /// Configures and initializes the LLama model.
        /// </summary>
        /// <param name="modelPath">Path to the LLama model file.</param>
        /// <param name="contextSize">Maximum length of chat memory context.</param>
        /// <param name="gpuLayerCount">Number of layers to offload to GPU.</param>
        public void Initialize(string modelPath, uint? contextSize = 1024, int gpuLayerCount = 5)
        {
            var parameters = new ModelParams(modelPath)
            {
                ContextSize = contextSize,
                GpuLayerCount = gpuLayerCount
            };

            _model = LLamaWeights.LoadFromFile(parameters);
            _context = _model.CreateContext(parameters);
            _executor = new InteractiveExecutor(_context);
        }

        /// <summary>
        /// Sets up a chat session with the given chat history.
        /// </summary>
        /// <param name="chatHistoryConfig">An optional configuration action to customize the initial chat history.</param>
        public void SetupChatSession(Action<ChatHistory>? chatHistoryConfig = null)
        {
            if (_executor == null)
            {
                throw new InvalidOperationException("LLamaController is not initialized. Call Initialize() first.");
            }

            var chatHistory = new ChatHistory();
            chatHistoryConfig?.Invoke(chatHistory);

            _session = new ChatSession(_executor, chatHistory);
        }

        /// <summary>
        /// Processes user input and generates responses from the AI.
        /// </summary>
        /// <param name="userInput">The input string from the user.</param>
        /// <param name="maxTokens">Maximum number of tokens to generate in the response.</param>
        /// <param name="antiPrompts">A list of anti-prompts to stop generation.</param>
        /// <returns>A streaming asynchronous response from the AI.</returns>
        public async IAsyncEnumerable<string> ProcessUserInputAsync(string userInput, int maxTokens = 256, List<string>? antiPrompts = null)
        {
            if (_session == null)
            {
                throw new InvalidOperationException("Chat session is not set up. Call SetupChatSession() first.");
            }

            var inferenceParams = new InferenceParams
            {
                MaxTokens = maxTokens,
                AntiPrompts = antiPrompts ?? new List<string> { "User:" }
            };

            await foreach (var response in _session.ChatAsync(new ChatHistory.Message(AuthorRole.User, userInput), inferenceParams))
            {
                yield return response;
            }
        }

        /// <summary>
        /// Cleans up resources used by LLamaController.
        /// </summary>
        public void Dispose()
        {
            _session = null;
            _executor = null;
            _context?.Dispose();
            _model?.Dispose();
        }
    }
}
