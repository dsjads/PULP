

package net.sf.zipme;
import java.io.IOException;
import java.io.InputStream;


public class InflaterInputStream extends InputStream {

  protected InputStream in;

  protected Inflater inf;

  protected byte[] buf;

  protected int len;
  private byte[] onebytebuffer=new byte[1];

  public InflaterInputStream(  InputStream in){
    this(in,new Inflater(),4096);
  }

  public InflaterInputStream(  InputStream in,  Inflater inf){
    this(in,inf,4096);
  }

  public InflaterInputStream(  InputStream in,  Inflater inf,  int size){
    this.in=in;
    if (in == null)     throw new NullPointerException("in may not be null");
    if (inf == null)     throw new NullPointerException("inf may not be null");
    if (size < 0)     throw new IllegalArgumentException("size may not be negative");
    this.inf=inf;
    this.buf=new byte[size];
  }

  public int available() throws IOException {
    if (inf == null)     throw new IOException("stream closed");
    return inf.finished() ? 0 : 1;
  }

  public synchronized void close() throws IOException {
    if (in != null)     in.close();
    in=null;
  }

  protected void fill() throws IOException {
    if (in == null)     throw new ZipException("InflaterInputStream is closed");
    len=in.read(buf,0,buf.length);
    if (len < 0)     throw new ZipException("Deflated stream ends early.");
    inf.setInput(buf,0,len);
  }

  public int read() throws IOException {
    int nread=read(onebytebuffer,0,1);
    if (nread > 0)     return onebytebuffer[0] & 0xff;
    return -1;
  }

  public int read(  byte[] buf) throws IOException {
    return read(buf,0,buf.length);
  }

  public int read(  byte[] b,  int off,  int len) throws IOException {
    if (inf == null)     throw new IOException("stream closed");
    if (len == 0)     return 0;
    int count=0;
    while (true) {
      try {
        count=inf.inflate(b,off,len);
      }
 catch (      DataFormatException dfe) {
        throw new ZipException(dfe.getMessage());
      }
      if (count > 0)       return count;
      if (inf.needsDictionary() | inf.finished())       return -1;
 else       if (inf.needsInput())       fill();
 else       throw new Error("Don't know what to do");
    }
  }

  public long skip(  long n) throws IOException {
    if (inf == null)     throw new IOException("stream closed");
    if (n < 0)     throw new IllegalArgumentException();
    if (n == 0)     return 0;
    int buflen=(int)Math.min(n,2048);
    byte[] tmpbuf=new byte[buflen];
    long skipped=0L;
    while (n > 0L) {
      int numread=read(tmpbuf,0,buflen);
      if (numread <= 0)       break;
      n-=numread;
      skipped+=numread;
      buflen=(int)Math.min(n,2048);
    }
    return skipped;
  }
  public boolean markSupported(){
    return false;
  }
  public void mark(  int readLimit){
  }
  public void reset() throws IOException {
    throw new IOException("reset not supported");
  }
}
