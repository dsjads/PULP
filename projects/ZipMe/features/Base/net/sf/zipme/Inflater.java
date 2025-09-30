

package net.sf.zipme;


public class Inflater {
  private static final int CPLENS[]={3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,67,83,99,115,131,163,195,227,258};
  private static final int CPLEXT[]={0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0};
  private static final int CPDIST[]={1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,385,513,769,1025,1537,2049,3073,4097,6145,8193,12289,16385,24577};
  private static final int CPDEXT[]={0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13};
  private static final int DECODE_HEADER=0;
  private static final int DECODE_DICT=1;
  private static final int DECODE_BLOCKS=2;
  private static final int DECODE_STORED_LEN1=3;
  private static final int DECODE_STORED_LEN2=4;
  private static final int DECODE_STORED=5;
  private static final int DECODE_DYN_HEADER=6;
  private static final int DECODE_HUFFMAN=7;
  private static final int DECODE_HUFFMAN_LENBITS=8;
  private static final int DECODE_HUFFMAN_DIST=9;
  private static final int DECODE_HUFFMAN_DISTBITS=10;
  private static final int DECODE_CHKSUM=11;
  private static final int FINISHED=12;

  private int mode;

  private int readAdler;

  private int neededBits;
  private int repLength, repDist;
  private int uncomprLen;

  private boolean isLastBlock;

  private long totalOut;

  private long totalIn;

  private boolean nowrap;
  private StreamManipulator input;
  private OutputWindow outputWindow;
  private InflaterDynHeader dynHeader;
  private InflaterHuffmanTree litlenTree, distTree;

  public Inflater(){
    this(false);
  }

  public Inflater(  boolean nowrap){
    this.nowrap=nowrap;
    this.hook32();
    input=new StreamManipulator();
    outputWindow=new OutputWindow();
    mode=nowrap ? DECODE_BLOCKS : DECODE_HEADER;
  }

  public void end(){
    outputWindow=null;
    input=null;
    dynHeader=null;
    litlenTree=null;
    distTree=null;
  }

  public boolean finished(){
    return mode == FINISHED && outputWindow.getAvailable() == 0;
  }

  public int getRemaining(){
    return input.getAvailableBytes();
  }

  public int getTotalIn(){
    return (int)(totalIn - getRemaining());
  }

  public long getBytesRead(){
    return totalIn - getRemaining();
  }

  public int getTotalOut(){
    return (int)totalOut;
  }

  public long getBytesWritten(){
    return totalOut;
  }

  public int inflate(  byte[] buf) throws DataFormatException {
    return inflate(buf,0,buf.length);
  }

  public int inflate(  byte[] buf,  int off,  int len) throws DataFormatException {
    if (len == 0)     return 0;
    if (0 > off || off > off + len || off + len > buf.length)     throw new ArrayIndexOutOfBoundsException();
    int count=0;
    int more;
    do {
      if (mode != DECODE_CHKSUM) {
        more=outputWindow.copyOutput(buf,off,len);
        this.hook33(buf,off,more);
        off+=more;
        count+=more;
        totalOut+=more;
        len-=more;
        if (len == 0)         return count;
      }
    }
 while (decode() || (outputWindow.getAvailable() > 0 && mode != DECODE_CHKSUM));
    return count;
  }

  public boolean needsDictionary(){
    return mode == DECODE_DICT && neededBits == 0;
  }

  public boolean needsInput(){
    return input.needsInput();
  }

  public void reset(){
    mode=nowrap ? DECODE_BLOCKS : DECODE_HEADER;
    totalIn=totalOut=0;
    input.reset();
    outputWindow.reset();
    dynHeader=null;
    litlenTree=null;
    distTree=null;
    isLastBlock=false;
  }

  public void setDictionary(  byte[] buffer){
    setDictionary(buffer,0,buffer.length);
  }

  public void setDictionary(  byte[] buffer,  int off,  int len){
    if (!needsDictionary())     throw new IllegalStateException();
    this.hook34(buffer,off,len);
    outputWindow.copyDict(buffer,off,len);
    mode=DECODE_BLOCKS;
  }

  public void setInput(  byte[] buf){
    setInput(buf,0,buf.length);
  }

  public void setInput(  byte[] buf,  int off,  int len){
    input.setInput(buf,off,len);
    totalIn+=len;
  }

  private boolean decodeHeader() throws DataFormatException {
    int header=input.peekBits(16);
    if (header < 0)     return false;
    input.dropBits(16);
    header=((header << 8) | (header >> 8)) & 0xffff;
    if (header % 31 != 0)     throw new DataFormatException("Header checksum illegal");
    if ((header & 0x0f00) != (Deflater.DEFLATED << 8))     throw new DataFormatException("Compression Method unknown");
    if ((header & 0x0020) == 0) {
      mode=DECODE_BLOCKS;
    }
 else {
      mode=DECODE_DICT;
      neededBits=32;
    }
    return true;
  }

  private boolean decodeDict(){
    while (neededBits > 0) {
      int dictByte=input.peekBits(8);
      if (dictByte < 0)       return false;
      input.dropBits(8);
      readAdler=(readAdler << 8) | dictByte;
      neededBits-=8;
    }
    return false;
  }

  private boolean decodeHuffman() throws DataFormatException {
    int free=outputWindow.getFreeSpace();
    while (free >= 258) {
      int symbol;
switch (mode) {
case DECODE_HUFFMAN:
        while (((symbol=litlenTree.getSymbol(input)) & ~0xff) == 0) {
          outputWindow.write(symbol);
          if (--free < 258)           return true;
        }
      if (symbol < 257) {
        if (symbol < 0)         return false;
 else {
          distTree=null;
          litlenTree=null;
          mode=DECODE_BLOCKS;
          return true;
        }
      }
    try {
      repLength=CPLENS[symbol - 257];
      neededBits=CPLEXT[symbol - 257];
    }
 catch (    ArrayIndexOutOfBoundsException ex) {
      throw new DataFormatException("Illegal rep length code");
    }
case DECODE_HUFFMAN_LENBITS:
  if (neededBits > 0) {
    mode=DECODE_HUFFMAN_LENBITS;
    int i=input.peekBits(neededBits);
    if (i < 0)     return false;
    input.dropBits(neededBits);
    repLength+=i;
  }
mode=DECODE_HUFFMAN_DIST;
case DECODE_HUFFMAN_DIST:
symbol=distTree.getSymbol(input);
if (symbol < 0) return false;
try {
repDist=CPDIST[symbol];
neededBits=CPDEXT[symbol];
}
 catch (ArrayIndexOutOfBoundsException ex) {
throw new DataFormatException("Illegal rep dist code");
}
case DECODE_HUFFMAN_DISTBITS:
if (neededBits > 0) {
mode=DECODE_HUFFMAN_DISTBITS;
int i=input.peekBits(neededBits);
if (i < 0) return false;
input.dropBits(neededBits);
repDist+=i;
}
outputWindow.repeat(repLength,repDist);
free-=repLength;
mode=DECODE_HUFFMAN;
break;
default :
throw new IllegalStateException();
}
}
return true;
}

private boolean decodeChksum() throws DataFormatException {
while (neededBits > 0) {
int chkByte=input.peekBits(8);
if (chkByte < 0) return false;
input.dropBits(8);
readAdler=(readAdler << 8) | chkByte;
neededBits-=8;
}
this.hook35();
mode=FINISHED;
return false;
}

private boolean decode() throws DataFormatException {
switch (mode) {
case DECODE_HEADER:
return decodeHeader();
case DECODE_DICT:
return decodeDict();
case DECODE_CHKSUM:
return decodeChksum();
case DECODE_BLOCKS:
if (isLastBlock) {
if (nowrap) {
mode=FINISHED;
return false;
}
 else {
input.skipToByteBoundary();
neededBits=32;
mode=DECODE_CHKSUM;
return true;
}
}
int type=input.peekBits(3);
if (type < 0) return false;
input.dropBits(3);
if ((type & 1) != 0) isLastBlock=true;
switch (type >> 1) {
case DeflaterConstants.STORED_BLOCK:
input.skipToByteBoundary();
mode=DECODE_STORED_LEN1;
break;
case DeflaterConstants.STATIC_TREES:
litlenTree=InflaterHuffmanTree.defLitLenTree;
distTree=InflaterHuffmanTree.defDistTree;
mode=DECODE_HUFFMAN;
break;
case DeflaterConstants.DYN_TREES:
dynHeader=new InflaterDynHeader();
mode=DECODE_DYN_HEADER;
break;
default :
throw new DataFormatException("Unknown block type " + type);
}
return true;
case DECODE_STORED_LEN1:
{
if ((uncomprLen=input.peekBits(16)) < 0) return false;
input.dropBits(16);
mode=DECODE_STORED_LEN2;
}
case DECODE_STORED_LEN2:
{
int nlen=input.peekBits(16);
if (nlen < 0) return false;
input.dropBits(16);
if (nlen != (uncomprLen ^ 0xffff)) throw new DataFormatException("broken uncompressed block");
mode=DECODE_STORED;
}
case DECODE_STORED:
{
int more=outputWindow.copyStored(input,uncomprLen);
uncomprLen-=more;
if (uncomprLen == 0) {
mode=DECODE_BLOCKS;
return true;
}
return !input.needsInput();
}
case DECODE_DYN_HEADER:
if (!dynHeader.decode(input)) return false;
litlenTree=dynHeader.buildLitLenTree();
distTree=dynHeader.buildDistTree();
mode=DECODE_HUFFMAN;
case DECODE_HUFFMAN:
case DECODE_HUFFMAN_LENBITS:
case DECODE_HUFFMAN_DIST:
case DECODE_HUFFMAN_DISTBITS:
return decodeHuffman();
case FINISHED:
return false;
default :
throw new IllegalStateException();
}
}
protected void hook32(){
}
protected void hook33(byte[] buf,int off,int more) throws DataFormatException {
}
protected void hook34(byte[] buffer,int off,int len){
}
protected void hook35() throws DataFormatException {
}
}
