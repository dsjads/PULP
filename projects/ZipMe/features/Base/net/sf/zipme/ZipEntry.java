

package net.sf.zipme;
import java.util.Calendar;
import java.util.Date;


public class ZipEntry implements ZipConstants {
  private static final int KNOWN_SIZE=1;
  private static final int KNOWN_CSIZE=2;
  private static final int KNOWN_CRC=4;
  private static final int KNOWN_TIME=8;
  private static final int KNOWN_EXTRA=16;
  private static Calendar cal;
  private String name;
  private int size;
  private long compressedSize=-1;
  private int crc;
  private int dostime;
  private short known=0;
  private short method=-1;
  private byte[] extra=null;
  private String comment=null;
  int flags;
  public int offset;

  public static final int STORED=0;

  public static final int DEFLATED=8;

  public ZipEntry(  String name){
    int length=name.length();
    if (length > 65535)     throw new IllegalArgumentException("name length is " + length);
    this.name=name;
  }

  public ZipEntry(  ZipEntry e){
    this(e,e.name);
  }
  public ZipEntry(  ZipEntry e,  String name){
    this.name=name;
    known=e.known;
    size=e.size;
    compressedSize=e.compressedSize;
    crc=e.crc;
    dostime=e.dostime;
    method=e.method;
    extra=e.extra;
    comment=e.comment;
  }
  final void setDOSTime(  int dostime){
    this.dostime=dostime;
    known|=KNOWN_TIME;
  }
  final int getDOSTime(){
    if ((known & KNOWN_TIME) == 0)     return 0;
 else     return dostime;
  }

  public String getName(){
    return name;
  }

  public void setTime(  long time){
    Calendar cal=getCalendar();
synchronized (cal) {
      cal.setTime(new Date(time));
      dostime=(cal.get(Calendar.YEAR) - 1980 & 0x7f) << 25 | (cal.get(Calendar.MONTH) + 1) << 21 | (cal.get(Calendar.DAY_OF_MONTH)) << 16 | (cal.get(Calendar.HOUR_OF_DAY)) << 11 | (cal.get(Calendar.MINUTE)) << 5 | (cal.get(Calendar.SECOND)) >> 1;
    }
    this.known|=KNOWN_TIME;
  }

  public long getTime(){
    parseExtra();
    if ((known & KNOWN_TIME) == 0)     return -1;
    int sec=2 * (dostime & 0x1f);
    int min=(dostime >> 5) & 0x3f;
    int hrs=(dostime >> 11) & 0x1f;
    int day=(dostime >> 16) & 0x1f;
    int mon=((dostime >> 21) & 0xf) - 1;
    int year=((dostime >> 25) & 0x7f) + 1980;
    try {
      cal=getCalendar();
synchronized (cal) {
        cal.set(Calendar.YEAR,year);
        cal.set(Calendar.MONTH,mon);
        cal.set(Calendar.DAY_OF_MONTH,day);
        cal.set(Calendar.HOUR_OF_DAY,hrs);
        cal.set(Calendar.MINUTE,min);
        cal.set(Calendar.SECOND,sec);
        return cal.getTime().getTime();
      }
    }
 catch (    RuntimeException ex) {
      known&=~KNOWN_TIME;
      return -1;
    }
  }
  private static synchronized Calendar getCalendar(){
    if (cal == null)     cal=Calendar.getInstance();
    return cal;
  }

  public void setSize(  long size){
    if ((size & 0xffffffff00000000L) != 0)     throw new IllegalArgumentException();
    this.size=(int)size;
    this.known|=KNOWN_SIZE;
  }

  public long getSize(){
    return (known & KNOWN_SIZE) != 0 ? size & 0xffffffffL : -1L;
  }

  public void setCompressedSize(  long csize){
    this.compressedSize=csize;
  }

  public long getCompressedSize(){
    return compressedSize;
  }

  public void setCrc(  long crc){
    if ((crc & 0xffffffff00000000L) != 0)     throw new IllegalArgumentException();
    this.crc=(int)crc;
    this.known|=KNOWN_CRC;
  }

  public long getCrc(){
    return (known & KNOWN_CRC) != 0 ? crc & 0xffffffffL : -1L;
  }

  public void setMethod(  int method){
    if (method != ZipOutputStream.STORED && method != ZipOutputStream.DEFLATED)     throw new IllegalArgumentException();
    this.method=(short)method;
  }

  public int getMethod(){
    return method;
  }

  public void setExtra(  byte[] extra){
    if (extra == null) {
      this.extra=null;
      return;
    }
    if (extra.length > 0xffff)     throw new IllegalArgumentException();
    this.extra=extra;
  }
  private void parseExtra(){
    if ((known & KNOWN_EXTRA) != 0)     return;
    if (extra == null) {
      known|=KNOWN_EXTRA;
      return;
    }
    try {
      int pos=0;
      while (pos < extra.length) {
        int sig=(extra[pos++] & 0xff) | (extra[pos++] & 0xff) << 8;
        int len=(extra[pos++] & 0xff) | (extra[pos++] & 0xff) << 8;
        if (sig == 0x5455) {
          int flags=extra[pos];
          if ((flags & 1) != 0) {
            long time=((extra[pos + 1] & 0xff) | (extra[pos + 2] & 0xff) << 8 | (extra[pos + 3] & 0xff) << 16 | (extra[pos + 4] & 0xff) << 24);
            setTime(time);
          }
        }
        pos+=len;
      }
    }
 catch (    ArrayIndexOutOfBoundsException ex) {
    }
    known|=KNOWN_EXTRA;
    return;
  }

  public byte[] getExtra(){
    return extra;
  }

  public void setComment(  String comment){
    if (comment != null && comment.length() > 0xffff)     throw new IllegalArgumentException();
    this.comment=comment;
  }

  public String getComment(){
    return comment;
  }

  public boolean isDirectory(){
    int nlen=name.length();
    return nlen > 0 && name.charAt(nlen - 1) == '/';
  }

  public String toString(){
    return name;
  }

  public int hashCode(){
    return name.hashCode();
  }
}
