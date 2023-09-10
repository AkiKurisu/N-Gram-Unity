using System.Collections.Generic;
namespace Kurisu.NGram
{
    public class TokenizerReader
    {
        private readonly List<string> tokens = new();
        private readonly List<byte> indexs = new();
        public TokenizerReader()
        {
        }
        public TokenizerReader(string text)
        {
            ReadText(text);
        }
        public void ReadText(string text)
        {
            tokens.Clear();
            indexs.Clear();
            var lines = text.Split('\n');
            foreach (var line in lines)
            {
                int index;
                if ((index = line.IndexOf(':')) > 0)
                {
                    tokens.Add(line[..index]);
                    indexs.Add(byte.Parse(line[(index + 1)..]));
                }
            }
        }
        public byte GetIndex(string token)
        {
            return indexs[tokens.IndexOf(token)];
        }
        public string GetToken(byte index)
        {
            return tokens[indexs.IndexOf(index)];
        }
    }
}
