

package net.sf.zipme;


public class Adler32 implements Checksum {

  private static final int BASE=65521;
  private int checksum;

  public Adler32(){
    reset();
  }

  public void reset(){
    checksum=1;
  }

  public void update(  int bval){
    int s1=checksum & 0xffff;
    int s2=checksum >>> 16;
    s1=(s1 + (bval & 0xFF)) % BASE;
    s2=(s1 + s2) % BASE;
    checksum=(s2 << 16) + s1;
  }

  public void update(  byte[] buffer){
    update(buffer,0,buffer.length);
  }

  public void update(  byte[] buf,  int off,  int len){
    int s1=checksum & 0xffff;
    int s2=checksum >>> 16;
    while (len > 0) {
      int n=3800;
      if (n > len)       n=len;
      len-=n;
      while (--n >= 0) {
        s1=s1 + (buf[off++] & 0xFF);
        s2=s2 + s1;
      }
      s1%=BASE;
      s2%=BASE;
    }
    checksum=(s2 << 16) | s1;
  }

  public long getValue(){
    return (long)checksum & 0xffffffffL;
  }
}
