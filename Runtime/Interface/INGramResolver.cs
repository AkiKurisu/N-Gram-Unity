namespace Kurisu.NGram
{
    public interface INGramResolver
    {
        bool Success { get; }
        byte Result { get; }
        void Dispose();
        void Resolve(byte[] history, byte[] inference);
        void Resolve(byte[] history, byte[] inference, int historyStartIndex, int historyLength);
        void Complete();
    }
}
