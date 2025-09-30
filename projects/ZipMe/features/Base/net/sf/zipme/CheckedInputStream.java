

package net.sf.zipme;
import java.io.IOException;
import java.io.InputStream;


public class CheckedInputStream extends InputStream {

  protected InputStream in;

  public CheckedInputStream(  InputStream in,  Checksum sum){
    this.in=in;
    this.sum=sum;
  }

  public Checksum getChecksum(){
    return sum;
  }

  public int read() throws IOException {
    int x=in.read();
    if (x != -1)     sum.update(x);
    return x;
  }

  public int read(  byte[] buf) throws IOException {
    return read(buf,0,buf.length);
  }

  public int read(  byte[] buf,  int off,  int len) throws IOException {
    int r=in.read(buf,off,len);
    if (r != -1)     sum.update(buf,off,r);
    return r;
  }

  public long skip(  long n) throws IOException {
    if (n == 0)     return 0;
    int min=(int)Math.min(n,1024);
    byte[] buf=new byte[min];
    long s=0;
    while (n > 0) {
      int r=in.read(buf,0,min);
      if (r == -1)       break;
      n-=r;
      s+=r;
      min=(int)Math.min(n,1024);
      sum.update(buf,0,r);
    }
    return s;
  }

  public void mark(  int readlimit){
    in.mark(readlimit);
  }

  public boolean markSupported(){
    return in.markSupported();
  }

  public void reset() throws IOException {
    in.reset();
  }

  public int available() throws IOException {
    return in.available();
  }

  public void close() throws IOException {
    in.close();
  }

  private Checksum sum;
}
