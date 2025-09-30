

package net.sf.zipme;
import java.io.IOException;
import java.io.OutputStream;


public class CheckedOutputStream extends OutputStream {

  protected OutputStream out;

  public CheckedOutputStream(  OutputStream out,  Checksum cksum){
    this.out=out;
    this.sum=cksum;
  }

  public Checksum getChecksum(){
    return sum;
  }

  public void write(  int bval) throws IOException {
    out.write(bval);
    sum.update(bval);
  }

  public void write(  byte[] buf) throws IOException {
    write(buf,0,buf.length);
  }

  public void write(  byte[] buf,  int off,  int len) throws IOException {
    out.write(buf,off,len);
    sum.update(buf,off,len);
  }

  public void close() throws IOException {
    flush();
    out.close();
  }

  public void flush() throws IOException {
    out.flush();
  }

  private Checksum sum;
}
