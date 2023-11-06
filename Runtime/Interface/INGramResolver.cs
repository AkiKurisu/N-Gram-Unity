namespace Kurisu.NGram
{
    public interface INGramResolver
    {
        bool Success { get; }
        int Result { get; }
        int NGram { get; set; }
        void Dispose();
        void Resolve(int[] history, int[] inference);
        void Resolve(int[] history, int[] inference, int historyStartIndex, int historyLength);
        void Complete();
    }
}
