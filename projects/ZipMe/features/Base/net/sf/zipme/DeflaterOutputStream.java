

package net.sf.zipme;
import java.io.IOException;
import java.io.OutputStream;


public class DeflaterOutputStream extends OutputStream {

  protected OutputStream out;

  protected byte[] buf;

  protected Deflater def;

  public DeflaterOutputStream(  OutputStream out){
    this(out,new Deflater(),4096);
  }

  public DeflaterOutputStream(  OutputStream out,  Deflater defl){
    this(out,defl,4096);
  }

  public DeflaterOutputStream(  OutputStream out,  Deflater defl,  int bufsize){
    this.out=out;
    if (bufsize <= 0)     throw new IllegalArgumentException("bufsize <= 0");
    buf=new byte[bufsize];
    def=defl;
  }
}
